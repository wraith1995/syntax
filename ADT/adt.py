"""A module for parsing ASDL grammars into Python Class hierarchies.

Adopted from https://raw.githubusercontent.com/gilbo/atl/master/ATL/adt.py
And again from https://github.com/ChezJrk/asdl/blob/master/src/asdl_adt/adt.py.
"""
from asyncio import run_coroutine_threadsafe
import inspect  # noqa: F401
import json
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Collection, Mapping, Sequence
from collections.abc import Iterable as ABCIterable
from copy import copy, deepcopy
from dataclasses import Field, field, make_dataclass, replace
from itertools import chain
from tkinter.tix import Form
from types import ModuleType
import importlib.util
import os
import logging
import tempfile
from yapf.yapflib.yapf_api import FormatCode
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
from weakref import WeakValueDictionary

import asdl  # type: ignore
from fastcore.all import typedispatch  # type: ignore
try:
    from snake_egg._internal import PyVar  # type: ignore
except ImportError:
    PyVar = None

logger = logging.getLogger(__name__)

defaultsTy = Mapping[Union[str, type, Tuple[str, str], Tuple[str, type]], Any]


indent = "    "
#based on test_ueq grammar: Should I add this to test_ueq_grammar?
# class TypeChecker:
#     def __init__(self, adt):
#         self.adt = adt

#     def check(self, node):
#         node_type = type(node)
#         if node_type == Symbol:
#             return True
#         elif node_type == Const:
#             return isinstance(node.val, int)
#         elif node_type == Var:
#             return isinstance(node.name, str)
#         elif node_type == Add:
#             return self.check(node.lhs) and self.check(node.rhs)
#         elif node_type == Scale:
#             return isinstance(node.coeff, int) and self.check(node.e)
#         elif node_type == Eq:
#             return self.check(node.lhs) and self.check(node.rhs)
#         elif node_type == Conj or node_type == Disj:
#             return all(self.check(pred) for pred in node.preds)
#         elif node_type == Cases:
#             return isinstance(node.case_var, Symbol) and all(self.check(pred) for pred in node.cases)
#         else:
#             return False


class ADTOptions(NamedTuple):
    """Configuration structure for ADTs."""

    copyMethods: bool
    visit: bool
    mapper: bool
    setfunctions: bool
    shortcutInit: bool
    loop: bool
    other: dict
    pprint: bool
    walkers: bool


defaultOpts = ADTOptions(
    copyMethods=True,
    visit=True,
    mapper=True,
    setfunctions=True,
    shortcutInit=True,
    loop=True,
    other={},
    pprint=True,
    walkers=True,
)



class ADTCreationError(Exception):
    """Base exception for errors in creating an ADT."""

    pass


class GenericADTError(Exception):
    """Base exception for errors in using the ADT."""

    pass


class _AsdlAdtBase(ABC):
    @abstractmethod
    def __init__(self):
        raise AssertionError("Should be unreachable")


class _ProdBase(_AsdlAdtBase):
    pass


class _SumBase(_AsdlAdtBase):
    pass

#or am i supposed to put Sym and pred in this class
# class Sym(ABC):
#     @abstractmethod
#     def __init__(self):
#         pass

# class pred(ABC):
#     @abstractmethod
#     def __init__(self):
#         pass
#create a function that just creates a few of these questions in the output
    
class _AbstractAbstractVisitor(ABC):
    @abstractmethod
    def __getitem__(self, t: Tuple[type, type]) -> Callable:
        pass


class AbstractVisitor(_AbstractAbstractVisitor):
    """Abstract class for visitors."""

    def __getitem__(self, t: Tuple[type, type]) -> Callable:
        """Get Vistor function."""
        return typedispatch[t]


def _asdl_parse(str):
    parser = asdl.ASDLParser()
    module = parser.parse(str)
    return module


_builtin_checks = {"object": lambda x: x is not None}

_builtin_types = {
    "string": str,
    "str": str,
    "int": int,
    "object": object,
    "float": float,
    "bool": bool,
}


class field_data(NamedTuple):
    """Class for field of a class."""

    name: str
    seq: bool
    opt: bool
    ty: str
    chk: Callable
    hasDefault: bool
    default: Any


# def fdtypestr(fd: field_data, env) -> str:
#     """Represent a field type as a string."""
#     tyname = fd.ty
#     if env.isInterallyDefined(fd.ty):
#         tyname += "_type"
#     if fd.seq:
#         return "typing.Sequence[{0}]".format(tyname)
#     elif fd.opt:
#         return "typing.Optional[{0}]".format(tyname)
#     else:
#         return tyname


# def fdinitstr(fd: field_data, env) -> str:
#     """Represent type signature of an init."""
#     tystr = fdtypestr(fd, env)
#     start = "{0}:{1}".format(fd.name, tystr)
#     if fd.hasDefault:
#         start += " = ..."
#     return start


def build_field_cdata(
    outerName: str,
    f,
    externalTypes: Mapping[str, type],
    internalTypes: Mapping[str, type],
    checks: Mapping[str, Callable],
    defaults: defaultsTy,
) -> field_data:
    """Transform enviroment info to field_data."""
    if str(f.type) in internalTypes:
        ty = internalTypes[str(f.type)]
    elif str(f.type) in externalTypes:
        ty = externalTypes[str(f.type)]
    else:
        raise ADTCreationError("{1}: Type {0} not defined".format(f.type, f.name))
    default = None
    hasDefault = True
    if (outerName, f.name) in defaults:
        default = defaults[(outerName, f.name)]
    elif (outerName, ty) in defaults:
        default = defaults[(outerName, ty)]
    elif ty in defaults:
        default = defaults[ty]
    else:
        if f.opt:
            hasDefault = True
            if f.seq:
                default = []
            else:
                default = None
        else:
            hasDefault = False
    dumb: Callable = lambda _: True
    return field_data(
        f.name,
        f.seq,
        f.opt,
        ty.__name__,
        checks[str(f.type)] if str(f.type) in checks else dumb,
        hasDefault,
        default,
    )


class constructor_data(NamedTuple):
    """Represent a constructor."""

    sup: str
    fields: list[field_data]
    name: str
    minArgs: int
    maxArgs: int
    minSatisfy: list[field_data]


class ADTEnv:
    """An enviroment for building ADTs."""

    def __init__(
        self,
        name: str,
        adsl_adt: asdl.Module,
        external_types: Mapping[str, type],
        defaults: defaultsTy,
        checks: Mapping[str, Callable],
        egraphableTypes: Union[bool, Set[str]] = False,
        options: ADTOptions = defaultOpts,
    ):
        """Create an enviroment for building ADTs."""
        self.name = name
        self.options = options
        self.checks = checks
        self.defaults: defaultsTy = defaults  # (name, field), (name, type), type
        self.sumClass: type = type("sum_" + name, (_SumBase,), {})
        self.prodClass: type = type("prod_" + name, (_ProdBase,), {})
        self.superTypes: Dict[str, type] = dict()
        self.externalTypes = external_types
        self.constructorDataPre: Dict[str, Tuple[List[str], type]] = dict()
        self.constructorData: dict[str, constructor_data] = dict()
        self.egraphableTypes: Union[bool, Set[str]] = egraphableTypes
        self.old = adsl_adt
        self.typeCollections: Dict[str, List[str]] = dict()

        def fieldValidator(flds, name, names=set()) -> List[str]:
            for fld in flds:
                if fld.name in names:
                    raise ADTCreationError("In {0}, name conflict with {1}".format(name, fld.name))
                else:
                    names.add(fld.name)
            return names

        for name, ty in adsl_adt.types.items():
            if name in self.constructorData:
                raise ADTCreationError("{0} conflicts with another name already defined".format(name))
            if isinstance(ty, asdl.Product):
                typ = type(name, (self.prodClass,), {})
                self.superTypes[name] = typ
                myattrs: Set[str] = set()
                fieldValidator(ty.fields, name, names=myattrs)
                self.constructorDataPre[name] = (ty.fields, typ)
                self.typeCollections[name] = [name]
            elif isinstance(ty, asdl.Sum):
                typ = type(name, (self.sumClass,), {})
                self.superTypes[name] = typ
                myattrs = set()
                fieldValidator(ty.attributes, name, names=myattrs)
                self.typeCollections[name] = []
                for summand in ty.types:
                    if summand.name in self.constructorDataPre:
                        raise ADTCreationError("{0} conflicts with another name already defined".format(summand.name))
                    fieldValidator(summand.fields, summand.name, names=myattrs.copy())
                    self.constructorDataPre[summand.name] = (
                        summand.fields + ty.attributes,
                        typ,
                    )
                    self.typeCollections[name].append(summand.name)

            else:
                raise ADTCreationError("ASDL item not sum nor product.")
        for name, (fields, ty) in self.constructorDataPre.copy().items():
            fieldData: list[field_data] = [
                build_field_cdata(
                    name,
                    f,
                    self.externalTypes,
                    self.superTypes,
                    self.checks,
                    self.defaults,
                )
                for f in fields
            ]
            maxArgs = len(fieldData)
            minSet = list(filter(lambda x: not x.hasDefault, fieldData))
            minArgs = len(minSet)
            self.constructorData[name] = constructor_data(ty.__name__, fieldData, name, maxArgs, minArgs, minSet)
        self.allTypeNames = set().union(*[set(v) for v in self.typeCollections.values()])

    def isInterallyDefined(self, typ: str) -> bool:
        """Determine if a type is internally defined."""
        
        return issubclass(eval(typ), self.sumClass) or issubclass(eval(typ), self.prodClass)

    def isInternalSum(self, typ: type) -> bool:
        """Determine if a type is an internally defined sum type."""
        return issubclass(typ, self.sumClass)

    def isInternalProduct(self, typ: type) -> bool:
        """Determine if a type is an internally defined product type."""
        return issubclass(typ, self.prodClass)

    
    def generateImportsAndErrors(self):
        """Generate the Imports and Errors for an ADT"""
        
    def generateStubSimplified(self):
        stub_commands = [
            "from dataclasses import dataclass",
            "import abc",
            "import typing",
            "from typing import Any, Union",
            "from ADT import stamp",
            "from abc import ABC, abstractmethod",
            "import collections.abc",
            "from collections.abc import Collection, Mapping, Sequence",
            "from collections.abc import Iterable",
            "import collections",
            "from weakref import WeakValueDictionary"
        ]
        str_version= '\n'.join(stub_commands)
        return str_version

    def generateStub(self) -> str:
        """Generate stub file for an ADT."""
        stub_commands = [
            "from dataclasses import dataclass",
            "import abc",
            "import typing",
            "from typing import Any, Union",
            "from ADT import stamp",
        ]
        pyVarCS = ["try:",
                   indent + "from snake_egg._internal import PyVar",
                   "except ImportError:",
                   indent + "PyVar = None"]
        if PyVar is not None:
            stub_commands += pyVarCS
        stub_commands.append("__all__ = ['{0}']\n".format(",".join(self.define_all())))
        for name in self.superTypes:
            # stub_commands.append("{0}_type: typing.TypeAlias = {0}\n".format(name))
            if issubclass(self.superTypes[name], self.sumClass):
                names = ", ".join(self.typeCollections[name])
                stub_commands.append("{1}_type: typing.TypeAlias = typing.Union[{0}]\n".format(names, name))
                stub_commands.append("{1}: typing.TypeAlias = typing.Union[{0}]\n".format(names, name))
                stub_commands.append("class _{0}(abc.ABCMeta): ...".format(name))
            else:
                stub_commands.append("{0}_type: typing.TypeAlias = {0}\n".format(name))
        for name, _ in self.constructorData.items():
            stub_commands += self.generateClassStub(name)
        stub_commands.append("_Any: typing.TypeAlias = Union[{0}]".format(", ".join(self.allTypeNames)))
        return "\n".join(stub_commands)

    def define_all(self) -> List[str]:
        """List all things exported by the created module."""
        # FIXME: Somehow do this automatically?
        all_defs = set([])
        for t in self.superTypes:
            if t not in all_defs:
                all_defs.add(t)
        for t in self.constructorData:
            if t not in all_defs:
                all_defs.add(t)
                if self.constructorData[t].maxArgs == 0:
                    all_defs.add("_" + t)
        all_defs.add("_Any")
        return list(all_defs)

    def useEgraph(self, ty: str) -> bool:
        """Check if egraph types are allowed."""
        if isinstance(self.egraphableTypes, bool):
            return self.egraphableTypes
        else:
            return ty in self.egraphableTypes

    def anyEgraph(self) -> bool:
        """Check if any egraph types are used."""
        if isinstance(self.egraphableTypes, bool):
            return self.egraphableTypes
        else:
            return len(self.egraphableTypes) > 0


def build_local_adt_errors(Err, x, cname, fieldName, targetType, opt):
    """Build errors used in ADT creation."""
    xt = type(x)
    badType = Err(
        """{0}.{1} does not have type
    {2} because a value {3}
    has type {4} and opt={5}""".format(
            cname, fieldName, targetType, x, xt, opt
        )
    )
    badCheck = Err(
        """{0}.{1} is not valid because
    {2} failed the
    check for type {3}""".format(
            cname, fieldName, x, targetType
        )
    )
    badSeq = Err(
        """{0}.{1} does not have type {2},
    and instead has type {3};
    we tried to convert the value, {4}, because it was a
    sequence or mapping,
    but this failed.""".format(
            cname, fieldName, targetType, xt, x
        )
    )
    badElem = Err(
        """{0}.{1} does not have type {2},
    and instead has type {3};
    we tried to convert the value, {4}, because it was the single
    correct type,
    but this failed.""".format(
            cname, fieldName, targetType, xt, x
        )
    )
    return badType, badSeq, badElem, badCheck

def get_source_code(obj):
    source_lines, _ = inspect.getsourcelines(obj)
    return ''.join(source_lines)

def save_type_source_to_file(typ, filename):
    source_code = get_source_code(typ)
    with open(filename, 'w') as file:
        file.write(source_code)

def get_import_statement(obj):
    module = inspect.getmodule(obj)
    if module is not None:
        module_path = inspect.getfile(module)
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        return f"from {module_name} import {obj.__name__}"
    
def addImports(fieldData,Err,cname,element_checker):
    imp=[]
    for fd in fieldData:
        imp.append(f"import {fd.ty.__qualname__}")
        imp.append(f"import {fd.chk.__qualname__}")
    strImp='\n'.join(imp)
    return strImp
        
def badElem(cname,fieldname,targetType,currentType,currentVal,Err):
    badElemErr= Err(
    """{0}.{1} does not have type {2},
    and instead has type {3};
    we tried to convert the value, {4}, because it was a
    sequence or mapping,
    but this failed.""".format(
            cname, fieldname, targetType, currentType, currentVal
        )
    )
    return badElemErr

def badCheck(cname,fieldname,x,targetType,Err):
    badCheck = Err(
        """{0}.{1} is not valid because
    {2} failed the
    check for type {3}""".format(
            cname, fieldname, x, targetType
        )
    )
    return badCheck


def build_post_init_str(fieldData,Err:type,cname,element_checker):
    """Build string version of post init function"""
    final=[]
    for fd in fieldData:
        fd_type= fd.ty 
        fieldname=fd.name
        val= fd.name
        tyname=fd.ty.__name__
        
        if fd.seq:
           
            new= f"""
                    val= {val}
                    tyname = {tyname}
                    if isinstance(val,Iterable):
                        val=tuple(val)
                        #check each element in the sequence
                        for x in val: 
                            try: 
                                vals.append(x)
                            except BaseException:
                                badElemErr= badElem({cname},{fieldname},{fd_type},{type}(x),x,{Err})
                                raise badElemErr
                            if not (val is None and opt) and not chk: 
                                badCheckErr= badCheck({cname},{fieldname},x,{fd_type},{Err})
                                raise badCheckErr
                        valsp = tuple(vals)
                        object.__setattr__(self, name, valsp)
                    """
            final.append(new)
        elif fd.opt:
            new= f"""
                val= {val}
                tyname = {tyname}                
                try:
                    object.__setattr__(self, name, val)
                except BaseException:
                    badElemErr= badElem({cname},{fieldname},{fd_type},{type}(val),val,{Err})
              """
            final.append(new)
              
        else:
            pass
    conc_str= '\n'.join(final)
    def __post_init__(self):
        return conc_str


        #make a string that is the same as the post init and concatenate
        #do I need to write this to output before calling the inner function or what
    
    #make sure there is code for the Errors (just put this at the top of the file)
    
    """
    1. Loop through the fields and make sure we important everytthing we need
    (for now, let's not do this)
    2. For each field, we generate a string that: acceses the field and based on (seq, opt), decides how to check it
    2.1. if not seq/not opt, you can do nothing - the idea is later we will check types and run checks
    2.2. If it is seq, then you must conver to a tuple and write back
    2.3. If it is optional, it can be none or the type.
    3. Yay!
    4. Later, we can deal with adding in the exceptiosn and adding in the calls to check or type that are externally type
    """



def build_visitor_accept(fieldData):
    """Build a simple visitor acceptor."""

    def accept(self, visit: _AbstractAbstractVisitor):
        visit[(type(visit), type(self))](visit, self)  # Take self paramter and the larget object
        for fd in fieldData:
            test = False
            try:
                test = callable(visit[type(visit), fd.ty])
            except BaseException:
                continue
            if not test:
                continue
            nxt = getattr(self, fd.name)
            if fd.opt:
                if nxt is None:
                    continue
                else:
                    nxt.accept(visit)
            elif fd.seq:
                for item in nxt:
                    item.accept(visit)
            else:
                nxt.accept(visit)

    return accept

def _build_classes_test(
    env: ADTEnv,
    slots: bool = False,
):
    dataclasses=[]
    abc_classes=[]
    sup={}
    names= {}
    my_dict={}
    dataclass_string=f"classdict_Sym:WeakValueDictionary=WeakValueDictionary({my_dict}) \n@dataclass(frozen={True}) \nclass Sym: \n name:str"
    formatted_code=FormatCode(dataclass_string)
    abc_classes.append(formatted_code[0])
    for name, data in env.constructorData.items():
        # breakpoint()

        if name:
            # if name not in names:
            #     names[name]=1
            #     abc_classes.append( 
            #     f"""class {name}(ABC):
            #         @abstractmethod
            #         def __init__(self):
            #             pass """ )
            if data.sup:
                if data.sup not in sup and data.sup not in names and data.sup!=name:
                    sup[data.sup]=1;
                    abc_classes.append( 
                        f"""class {data.sup}(ABC):
                            @abstractmethod
                            def __init__(self):
                                pass """ )
        #breakpoint()
                    

        dataclasses.append(build_dc_test(env,name,data.sup,data.fields,slots=slots)) #should return a string
  
    abc_classes_str= "\n".join(abc_classes)
    formatted_code_str=FormatCode(abc_classes_str)
    # dataclasses= [formatted_code]+dataclasses
    all_dataclasses_str="\n".join(dataclasses)
    formatted_code_dataclass= FormatCode(all_dataclasses_str)
    formatted_code_dataclass_final=formatted_code_dataclass[0]
    formatted_code_str_final= formatted_code_str[0]
    
    return formatted_code_str_final+formatted_code_dataclass_final

#Do I need to make this recursive? This would be easier using visitor pattern imo (ask about this)
def  generate_post_init_class(cname):
    post_init=[]
    if cname == "problem":
        post_init.append(f"\tdef __post_init__(self):")
        return f"""
def __post_init__(self):
    assert isinstance(self.holes,Iterable)
    for x in self.holes:
        if not isinstance(x,Sym):
            assert isinstance(x,str)
            xp= Sym(x)
            assert isinstance(xp,Sym) 
            
    assert isinstance(self.knowns,Iterable)
    for x in self.holes:
        if not isinstance(x,Sym):
            assert isinstance(x,str)
            xp= Sym(x)
            assert isinstance(xp,Sym) 
    assert isinstance(self.preds,Iterable)
    for x in self.preds:
        if isinstance(x,Conj) or isinstance(x,Disj):
            x.__post_init__()
                
        if isinstance(x,Cases):
            assert(isinstance(x.case_var,Sym))
            x.cases.__post_init__()
        
        if isinstance(x,Eq):
            assert(isinstance(x.lhs,expr))
            x.lhs.__post_init__()
            assert(isinstance(x.rhs,expr))
            x.rhs.__post_init__()
            
        if not isinstance(x,pred):
            xp= pred(x)
            assert isinstance(xp,pred) 
    
    """
    if cname=="Conj" or cname=="Disj":
        return f"""
def __post_init__(self):
    assert isinstance(self.preds,Iterable)
    for x in self.preds:
        if isinstance(x,Conj) or isinstance(x,Disj):
            x.__post_init__()
                
        if isinstance(x,Cases):
            assert(isinstance(x.case_var,Sym))
            x.cases.__post_init__()
        
            
        if isinstance(x,Eq):
            assert(isinstance(x.lhs,expr))
            x.lhs.__post_init__()
            assert(isinstance(x.rhs,expr))
            x.rhs.__post_init__()
            
        if not isinstance(x,pred):
            
            xp= pred(x)
            assert isinstance(xp,pred) 
    """
    
    if cname== "Cases":
        return f"""
def __post_init__(self):
    if not isinstance(self.case_var,Sym):
        assert(isinstance(self.case_var,str))
        xp= Sym(case_var)
        assert isinstance(xp,case_var)
    assert isinstance(self.cases,Iterable)
    for x in self.cases:
        if isinstance(x,Conj) or isinstance(x,Disj):
            x.__post_init__()
                
        if isinstance(x,Cases):
            assert(isinstance(x.case_var,Sym))
            x.cases.__post_init__()
        
            
        if isinstance(x,Eq):
            assert(isinstance(x.lhs,expr))
            x.lhs.__post_init__()
            assert(isinstance(x.rhs,expr))
            x.rhs.__post_init__()
            
        if not isinstance(x,pred):
    
            xp= pred(x)
            assert isinstance(xp,pred)
             
    """
    
    if cname=="Eq":
        return f"""
def __post_init__(self):
    if isinstance(self.lhs,Const):
        assert(isinstance(self.lhs.val,int))
        
    if isinstance(self.lhs,Var):
        assert(isinstance(self.lhs.name,Sym))
        
    if isinstance(self.lhs,Add):
        self.lhs.lhs.__post_init__()
        self.lhs.rhs.__post_init__()
        
    if isinstance(self.lhs,Scale):
        assert(isinstance(self.lhs.coeff,int))
        self.lhs.e.__post_init__()
        
    if isinstance(self.rhs,Const):
        assert(isinstance(self.rhs.val,int))
        
    if isinstance(self.rhs,Var):
        assert(isinstance(self.rhs.name,Sym))
        
    if isinstance(self.rhs,Add):
        self.rhs.lhs.__post_init__()
        self.rhs.rhs.__post_init__()
        
    if isinstance(self.rhs,Scale):
        assert(isinstance(self.rhs.coeff,int))
        self.rhs.e.__post_init__()
        
    
    if not isinstance(self.lhs,expr):
        xp= expr(self.lhs)
        assert isinstance(xp,case_var)
        
    if not isinstance(self.rhs,expr):
        xp= expr(self.rhs)
        assert isinstance(xp,case_var)
            
    """
    
    if cname== "Const":
        # post_init.append(f"\tdef __post_init__(self):")
        # post_init.append(f"\t\tassert(isinstance(self.val,int))")
        # return "\n".join(post_init)
        return f"""
def __post_init__(self): 
    assert(isinstance(self.val,int))
        
        """
    if cname== "Var":
        return f"""
def __post_init__(self): 
    assert(isinstance(self.name,Sym))
        """
        
    if cname== "Add":
        return f"""
def __post_init__(self):
    if isinstance(self.lhs,Const):
        assert(isinstance(self.lhs.val,int))
        
    if isinstance(self.lhs,Var):
        assert(isinstance(self.lhs.name,Sym))
        
    if isinstance(self.lhs,Add):
        self.lhs.lhs.__post_init__()
        self.lhs.rhs.__post_init__()
        
    if isinstance(self.lhs,Scale):
        assert(isinstance(self.lhs.coeff,int))
        self.lhs.e.__post_init__()
        
    if isinstance(self.rhs,Const):
        assert(isinstance(self.rhs.val,int))
        
    if isinstance(self.rhs,Var):
        assert(isinstance(self.rhs.name,Sym))
        
    if isinstance(self.rhs,Add):
        self.rhs.lhs.__post_init__()
        self.rhs.rhs.__post_init__()
        
    if isinstance(self.rhs,Scale):
        assert(isinstance(self.rhs.coeff,int))
        self.rhs.e.__post_init__()
        
    
    if not isinstance(self.lhs,expr):
        xp= expr(self.lhs)
        assert isinstance(xp,case_var)
        
    if not isinstance(self.rhs,expr):
        xp= expr(self.rhs)
        assert isinstance(xp,case_var)
            
    """
    
    if cname=="Scale":
         return f"""
def __post_init__(self):
    assert(isinstance(self.coeff,int))
    assert(isinstance(self.e,(Const,Var,Add,Scale)))
    
    if isinstance(self.e,Const):
        assert(isinstance(self.e.val,int))
        
    if isinstance(self.e,Var):
        assert(isinstance(self.e.name,Sym))
        
    if isinstance(self.e,Add):
        self.e.lhs.__post_init__()
        self.e.rhs.__post_init__()
        
    if isinstance(self.e,Scale):
        assert(isinstance(self.e.coeff,int))
        self.e.e.__post_init__()
        """
        
# def fdtypestr(fd: field_data, env) -> str:
#     """Represent a field type as a string."""
#     tyname = fd.ty
#     if env.isInterallyDefined(fd.ty):
#         tyname += "_type"
#     if fd.seq:
#         return "typing.Sequence[{0}]".format(tyname)
#     elif fd.opt:
#         return "typing.Optional[{0}]".format(tyname)
#     else:
#         return tyname
        
        
def build_dc_test(
    env:ADTEnv,
    # Err:type,
    cname: str,
    parent: str,
    fieldData: list[field_data],
    slots: bool = True,
    ):
    
    Err= type("Error",(Exception,),{})
    
    post_init= generate_post_init_class(cname)
    formatted_post_init= FormatCode(post_init)
    
    #go thru field data, and ask if type is defined within the system: if it's not, we call getsource and append text to top of file
    #check if defined within the system by env.isInternallyDefined
    def fieldp(x) -> Field:
        tmp = field(default_factory=x) if callable(x) else field(default=x)
        assert isinstance(tmp, Field)
        return tmp

    bf = field(default=("___" + cname + "__"), init=False, repr=False) #dataclass field 
    assert isinstance(bf, Field)
    extra: List[Tuple[str, Type[Any], Field]] = [("___" + cname + "__", str, bf)]
    # fields: List[Union[Tuple[str, Type[Any], Field], Tuple[str, Type[Any]]]] = [
    #     (fd.name, fd.ty) if not fd.hasDefault else (fd.name, fd.ty, fieldp(fd.default)) for fd in fieldData
    # ]
    
    #fields: List[Union[Tuple[str, Type[Any], Field], Tuple[str, Type[Any]]]] = []
    fields=[]
    for fd in fieldData:
        #breakpoint()
        typ=fd.ty
        if not fd.hasDefault:
            if fd.seq:
                if typ:
                    # breakpoint()
                    fields.append((fd.name, f"Iterable[{typ}]"))
                else:
                    fields.append((fd.name, None))
        
            # Check if the field is optional
            elif fd.opt:
                if typ:
                    fields.append((fd.name, f"typing.Optional[{typ}]"))
                else:
                    fields.append((fd.name, None))
            
            else:
                if typ:
                    fields.append((fd.name, typ))
                else:
                    fields.append((fd.name, None))
           
        else:
            if fd.seq:
                fields.append((fd.name,f"Iterable[{typ}]",fieldp(fd.default)))
            elif fd.opt:
                fields.append((fd.name,f"typing.Optional[{typ}]",fieldp(fd.default)))
            else: 
                fields.append((fd.name,fd.ty,fieldp(fd.default)))
            
    fields += extra
        #check if fd.seq is true (if it is then make an Iterable type)
        #check if fd.opt is true (and if it is then type is Optional[Type])
        
    #make a string that outputs the corresponding dataclass @dataclass /n class name 
   
    field_strings = []
    for field_tuple in fields:
        if len(field_tuple) == 2:
            name, typ = field_tuple
            # If no specific type is defined, use 'typing.Any'
            typ_str = "'typing.Any'" if typ is None else typ
            if typ_str=="Iterable":
                typ_str=f"{typ}"

            # field_strings.append(f"\t  {name}: {typ_str}")
            
            field_strings.append(f"\t  {name}: {typ_str}")
            # else: #in this case get source code, and append it to the start of the file
            #     field_strings.append(f"\t  {name}:typing.Any")
            
        else:
            name, typ, field_instance = field_tuple
            # If no specific type is defined, use 'typing.Any'
            typ_str = "'typing.Any'" if typ is None else typ
            pass
            # Format the string with the field instance and type
            # {field_instance.default}
            # field_strings.append(f"\t  {name}: {typ_str} = {field_instance.default}({typ_str})")
    # Concatenate field strings with newline character
    fields_string = "\n".join(field_strings)
    # do I need to get source code for parent?
    my_dict={}
    # create post init 
    # inner_post_init=""
    # if cname=="problem":
        
    # # post_init= f"""
    # # def __post_init__(self):
        
    
    # # """
    if parent!=cname:
        
            dataclass_string=f"classdict_{cname}:WeakValueDictionary=WeakValueDictionary({my_dict}) \n@dataclass(frozen={True},slots={slots}) \nclass {cname}({parent}): \n {fields_string} \n{post_init}"
#         dataclass_string = f"""
# @dataclass(frozen=True, slots={slots})
# class {cname}({parent}):
# {fields_string}
#         {post_init}
# """
        #formatted_code=FormatCode(dataclass_string)
        #final=formatted_code[0]
    else:
        dataclass_string=f"classdict_{cname}:WeakValueDictionary=WeakValueDictionary({my_dict}) \n@dataclass(frozen={True},slots={slots}) \nclass {cname}: \n {fields_string} \n {post_init}"
#         dataclass_string=f"""
# @dataclass(frozen=True, slots={slots})
# class {cname}:
# {fields_string}
#         {post_init}
# """
        #formatted_code=FormatCode(dataclass_string)
        #final=formatted_code[0]
        
    return dataclass_string

    
def build_function_category_methods(fieldData, env, Err):
    """Build methods to treat functions as functions as sets."""

    def __contains__(self, other):
        if self == other:
            return True
        else:
            for fd in fieldData:
                temp = getattr(self, str(fd.ty))
                if env.isInterallyDefined(fd.ty):
                    if temp.__contains__(other):
                        return True
                    else:
                        pass
            return False

    def isdisjoint(self, other):
        if self in other:
            return False
        else:
            for fd in fieldData:
                temp = getattr(self, fd.name)
                if env.isInterallyDefined(fd.ty):
                    if temp.isdisjoint(other):
                        pass
                    else:
                        return False
            return True

    def __isomorphism__(x, y, defs, equiv):
        if env.isInterallyDefined(type(x)) and env.isInterallyDefined(type(y)):
            if x is y:
                raise Err("Ismorpmism of non-disjoint objects")
            elif type(x) != type(y):
                return False
            elif x in defs:
                return defs[x] == y
            else:
                for fd in fieldData:
                    temp1 = getattr(x, fd.name)
                    temp2 = getattr(y, fd.name)
                    if env.isInterallyDefined(fd.ty):
                        fine = temp1.__isomorphism__(temp2, defs, equiv)
                    else:
                        fine = type(x) == type(y) and equiv(x, y)
                    if not fine:
                        return fine
                    else:
                        pass
                defs[x] = y
                return True
        else:
            return type(x) == type(y) and equiv(x, y)

    def isomorphism(self, other, equiv=lambda _: True):
        if not env.isInterallyDefined(type(other)):
            raise Err("Isomorphism not defined on external types")

        mapper = OrderedDict()
        if self.__isomorphism__(other, mapper, equiv):
            return mapper
        else:
            return None

    return isdisjoint, isomorphism, __isomorphism__


def build_element_copy_methods(fieldData, env):
    """Build methods to copy elements."""

    def mcopy(
        self,
        deep=False,
        copies={},
        ignore=set(),
        onlyCopies=False,
        copyInternal=True,
        copyExternal=False,
    ):
        if self in copies:
            return copies[self]
        if any(issubclass(type(self), ig) for ig in ignore):
            return self
        rep = {}
        for fd in fieldData:
            ty = fd.ty
            if ty in ignore:
                continue
            else:
                temp = getattr(self, fd.name)
                if temp is None:
                    continue
                temp_c = None
                if env.isInterallyDefined(ty) and copyInternal:
                    lst = [temp]
                    if fd.seq:
                        lst = temp
                    res = []
                    for item in lst:
                        if item in copies:
                            res.append(copies[item])
                        else:
                            item_c = item.__copy__(
                                deep=deep,
                                copies=copies,
                                ignore=ignore,
                                onlyCopies=onlyCopies,
                                copyInternal=copyInternal,
                                copyExternal=copyExternal,
                            )
                            copies[item] = item_c
                            res.append(item_c)
                    if fd.seq:
                        temp_c = tuple(res)
                    else:
                        temp_c = res[0]
                elif copyExternal:
                    lst = [temp]
                    if fd.seq:
                        lst = temp
                    res = []
                    for lcopy in lst:
                        res.append(deepcopy(lcopy) if deep else copy(lcopy))
                    if fd.seq:
                        temp_c = tuple(res)
                    else:
                        temp_c = res[0]
                else:
                    pass
                if temp_c is None:
                    continue
                rep[fd.name] = temp_c
        if len(rep) == 0 and onlyCopies:
            return self
        return replace(self, **rep)

    def update(self, **kwargs):
        return replace(self, **kwargs)

    def dcopy(self):
        return mcopy(self, deep=True)

    return mcopy, update, dcopy

    # FIXME: We clearly need types of iterations for this.
    # Iterate over the internal definitions vs over the external definitions.
    # FIXME: I should not be using fields here I think?

    # Depth vs Bredth
    # Self: first or last (pre order vs post order)
    # Emit names?
    # emit nones?
    # Emit dups?
    # order children?
    # Filter children? (internal, external, ...)
    # Flattening or no flattening?

    # Questions are: order, emission, consideration
    # What order?
    # When do I look in something?
    # When do I emit it?

    # Map, fold, filter, iter


def build_element_iteration_methods(Err, fieldData, env):
    """Add methods to iterate through elements."""

    def toMapping(mapper: Union[Mapping, Callable]) -> Mapping:
        if callable(mapper):
            raise Err("Bad use of map with Callable!")
        else:
            return mapper

    def map(self, mapper: Union[Mapping, Callable], unionSeq: bool = False):
        if not callable(mapper) and self in mapper:
            if callable(mapper):
                return mapper(self)
            else:
                return mapper[self]
        rep = {}
        for fd in fieldData:
            test = env.isInterallyDefined(fd.ty)
            temp = getattr(self, fd.name)
            iters = []
            if fd.seq:
                if unionSeq and not callable(mapper) and temp in mapper:
                    rep[fd.name] = mapper[temp]
                    continue
                else:
                    iters = temp
            reps = [
                toMapping(mapper)[x] if not callable(mapper) and x in mapper else (x.map(mapper) if test else x)
                for x in iters
            ]
            if not fd.seq:
                rep[fd.name] = reps[0]
            else:
                rep[fd.name] = reps
        return replace(self, **rep)

    def __iter__(self, internal: bool = True):
        yield self
        # go over fields that have interally defined types and iterate though
        # if they are lists, iterate through them tooo
        # make sure nothing is None.
        nexts = chain(
            *[
                (getattr(self, fd.name).loop() if not fd.seq else chain(*([x.loop() for x in getattr(self, fd.name)])))
                for fd in fieldData
                if (not internal or env.isInterallyDefined(fd.ty)) and getattr(self, fd.name) is not None  # noqa: W503
            ]
        )
        yield from nexts

    return __iter__, map

def build_err_str(mod,Err,env,cname):
    def element_checker(
        cname: str,
        fieldName: str,
        targetType: type,
        chk: Callable,
        opt: bool,
        x: Any,
    ):
        #Do I need mod?
        str_to_ret=[]
        
        xt = type(x)
        badType = Err(
            """{0}.{1} does not have type
        {2} because a value {3}
        has type {4} and opt={5}""".format(
                cname, fieldName, targetType, x, xt, opt
            )
        )
        badCheck = Err(
            """{0}.{1} is not valid because
        {2} failed the
        check for type {3}""".format(
                cname, fieldName, x, targetType
            )
        )
        badSeq = Err(
            """{0}.{1} does not have type {2},
        and instead has type {3};
        we tried to convert the value, {4}, because it was a
        sequence or mapping,
        but this failed.""".format(
                cname, fieldName, targetType, xt, x
            )
        )
        badElem = Err(
            """{0}.{1} does not have type {2},
        and instead has type {3};
        we tried to convert the value, {4}, because it was the single
        correct type,
        but this failed.""".format(
                cname, fieldName, targetType, xt, x
            )
        )
        
        str_to_ret.append(str(badCheck))
        str_to_ret.append(str(badElem))
        str_to_ret.append(str(badSeq))
        str_to_ret.append(str(badType))
        #only add the errors to the top of the file 
        str_to_ret.append(f"tyname={targetType.__name__}")
        
        #NOTE: ask if I can convert type callable to string just by putting brackets around it 
        #NOTE: DO I even need anything other than decl of badErr
        res=f"""
        singleType: Optional[type] = None
      
        convert = False
        if {x} is None and {opt}:
            return (False, None)
        if True:
                if singleType is not None:
                    # CHECK: How does opt interact with this case?
                    # CHECK: How does this interact with seq case?
                    try:
                        x = getattr(mod, tyname)({x})
                        convert = True
                    except BaseException:
                        raise badElem
                elif {x} is None and {opt}: # case where it is optional
                    pass
                else: #something has gone wrong with the typing 
                    raise badType
        else:
            pass

        if not ({x} is None and {opt}) and not {chk(x)}:
            raise badCheck
        else:
            return (convert, {x}) """
            
        res_formatted=FormatCode(res)
        str_to_ret.append(res_formatted[0])
        final_str= '\n'.join(str_to_ret)
        formatted_final_str=FormatCode(final_str)
        return formatted_final_str[0]
        
        
        
def build_element_check(mod, Err, env, cname):
    """Add method to check arguments and convert as needed."""
    
    #optional, sequence, or an item (check if its either of these)

    def element_checker(
        cname: str,
        fieldName: str,
        targetType: type,
        chk: Callable,
        opt: bool,
        x: Any,
    ):
        badType, badSeq, badElem, badCheck = build_local_adt_errors(Err, x, cname, fieldName, targetType, opt)

        tyname = targetType.__name__
        singleType: Optional[type] = None
      
     

        convert = False
        if x is None and opt:
            return (False, None)
        if True:
          
                if singleType is not None:
                    # CHECK: How does opt interact with this case?
                    # CHECK: How does this interact with seq case?
                    try:
                        x = getattr(mod, tyname)(x)
                        convert = True
                    except BaseException:
                        raise badElem
                elif x is None and opt: # case where it is optional
                    pass
                else: #something has gone wrong with the typing 
                    raise badType
        else:
            pass

        if not (x is None and opt) and not chk(x):
            raise badCheck
        else:
            return (convert, x)

    return element_checker




def build_new(memoize, classdict):
    """Make a new function for an ADT Entry."""

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        cls.__init__(obj, *args, **kwargs)
        # build the data class to check if it exists.
        # Hope this is gc'd quickly.
        if memoize and (obj in classdict):
            return classdict[(obj)]
        elif memoize:
            classdict[obj] = obj
            return obj
        else:
            return obj

    return __new__

#just need to call eval on generated string to get dict back
def build_new_str(memoize,classdict):
    """Represent new function as a string """
    classdict_str=repr(classdict)
    return f"""
    def __new__(cls, *args, **kwargs):
            obj = object.__new__(cls)
            cls.__init__(obj, *args, **kwargs)
            # build the data class to check if it exists.
            # Hope this is gc'd quickly.
            if {memoize} and (obj in {classdict_str}):
                return {classdict_str}[(obj)]
            elif {memoize}:
                {classdict_str}[obj] = obj
                return obj
            else:
                return obj

    return __new__
"""


def _build_classes(
    asdl_mod: asdl.Module,
    env: ADTEnv,
    memoize: Set[str],
    namespace_injector: Optional[Callable] = None,
    slots: bool = False,
    visitor: bool = True,
):
    mod = ModuleType(asdl_mod.name)
    Err: type = type(asdl_mod.name + "Error", (Exception,), {})
    setattr(mod, "__err__", Err)
    for name, ty in env.superTypes.items():
        setattr(mod, "_" + name, ty)
    for name, data in env.constructorData.items():
        dc = build_dc(
            env,
            data.name,
            data.sup,
            data.fields,
            Err,
            mod,
            memoize=name in memoize,
            namespace_injector=namespace_injector,
            slots=slots,
            visitor=visitor,
        )
        match dc:
            case (val, dc):
                setattr(mod, name, val)
                setattr(mod, "_" + name, dc)
            case _:
                setattr(mod, name, dc)
    for name, tys in env.typeCollections.items():
        tysList: List[Type] = [getattr(mod, name) for name in tys]
        setattr(mod, name, Union[tuple(tysList)])
    allTypeNames = env.allTypeNames
    allTypes = tuple([getattr(mod, name) for name in allTypeNames])
    setattr(mod, "_Any", Union[allTypes])
    setattr(mod, "__all__", env.define_all())
    return mod


def ADT(
    asdl_str: str,
    ext_types: Mapping[str, type] = {},
    ext_checks: Mapping[str, Callable] = {},
    defaults: defaultsTy = {},
    memoize: Union[Collection[str], bool] = True,
    slots: bool = False,
    visitor: bool = False,
    stubfile: Optional[str] = None,
    egraphableTypes: Union[bool, Set[str]] = False,
    options: ADTOptions = defaultOpts,
) -> ModuleType:
    r"""ADT converts an ASDL grammar into a Python Module.

    The returned module will contain one class for every ASDL type
    declared in the input grammar, and one (sub-)class for every
    constructor in each of those types.  These constructors will
    type-check objects on construction to ensure conformity with the
    given grammar.

    ASDL Syntax
    =================
    The grammar of ASDL follows this BNF::

        module      ::= "module" Id "{" [definitions] "}"
        definitions ::= { TypeId "=" type }
        type        ::= product | sum
        product     ::= fields ["attributes" fields]
        fields      ::= "(" { field, "," } field ")"
        field       ::= TypeId ["?" | "*"] [Id]
        sum         ::= constructor { "|" constructor } ["attributes" fields]
        constructor ::= ConstructorId [fields]

    Parameters
    =================
    asdl_str : str
        The ASDL definition string
    ext_checks : dict of functions, optional
        Type-checking functions for all external (undefined) types
        that are not "built-in".
        "built-in" types, and corresponding Python types are
        *   'object' - (anything except None)
    ext_types : dict of types, required
        Dictionary of external types to check against.

    Returns
    =================
    module
        A newly created module with classes for each ASDL type and constructor

    Example
    =================
    ::

        PolyMod = ADT(\"\"\" module PolyMod {
            expr = Var   ( id    name  )
                 | Const ( float val   )
                 | Sum   ( expr* terms )
                 | Prod  ( float coeff, expr* terms )
                 attributes( string? tag )
        }\"\"\", {
            "id" : lambda x: type(x) is str and str.isalnum(),
        })
    """
    asdl_ast = _asdl_parse(asdl_str)
    assert isinstance(asdl_ast, asdl.Module)

    env = ADTEnv(
        asdl_ast.name,
        asdl_ast,
        _builtin_types | ext_types,
        defaults,
        _builtin_checks | ext_checks,
        egraphableTypes=egraphableTypes,
        options=options,
    )
    
    
    classes= env.superTypes
    #need to create an abc class for each supertype
    if memoize is True:
        memoize = set(env.constructorData.keys())
    elif memoize is False:
        memoize = set()
    elif isinstance(memoize, set):
        pass
    else:
        raise ADTCreationError("Memoization should be a set or Bool")
    
    all_dataclasses_str= _build_classes_test(env,slots=slots)
    #make the name something you can parametrize
    stub= env.generateStubSimplified()
    
    badCheckSource=FormatCode(inspect.getsource(badCheck))
    badElemSource=FormatCode(inspect.getsource(badElem))
    
 
    
    # err_code= build_err_str(mod,Err,env,cname)

    
    # with open("dataclass_test_str.py",'w') as f:
    #    f.write(stub)
    tempFile = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    logging.info("Writing to {0}".format(tempFile.name))
    with tempFile as t:
        t.write(stub)
        t.write('\n')
        t.write(badCheckSource[0])
        t.write('\n')
        t.write(badElemSource[0])
        t.write('\n')
   
        t.write(all_dataclasses_str)
    
    logging.info("Wrote to {0}".format(tempFile.name))
    filePath = tempFile.name
    spec = importlib.util.spec_from_file_location(asdl_ast.name, filePath)
    assert spec is not None
    file = importlib.util.module_from_spec(spec)
    assert file is not None
    assert spec.loader is not None
    spec.loader.exec_module(file)
    return file

    # with open("dataclass_test_str.py", 'w') as f:
    #     f.write(all_dataclasses_str)
        
    # print("MAKES IT HERE")
        
    # mod=create_module_from_file("dataclass_test_str.py","dataclass_test_str")
    # if(mod is None):
    #     raise TypeError("module is None")
    # return mod

def create_module_from_file(file_path, module_name):
    with open(file_path, 'r') as file:
        code = file.read()

    spec = importlib.util.spec_from_loader(module_name, loader=None)
    if (spec is not None):
        module = importlib.util.module_from_spec(spec)
        exec(code, module.__dict__) #executes the code within modules namespace
        
        #This line adds the newly created module (module) to sys.modules, making it importable by its name (module_name). 
        # This step ensures that subsequent attempts to import the module will find it in the module cache.
        sys.modules[module_name] = module

        return module
    
    
    #need to turn this file into module

    
    

    # mod = _build_classes(asdl_ast, env, memoize=memoize, slots=slots, visitor=visitor)
    # # cache values in case we might want them
    # setattr(mod, "_ast", asdl_ast)  # noqa: B010
    # # mod._ast = asdl_ast
    # setattr(mod, "_defstr", asdl_str)  # noqa: B010
    # # mod._defstr = asdl_str
    # setattr(mod, "_env", env)
    # mod.__doc__ = f"""ASDL Module Generated by ADT\n\n"
    # f"Original ASDL description:\n{asdl_str}"""
    # if stubfile is not None:
    #     with open(stubfile, "w+") as f:
    #         text = env.generateStub("")
    #         f.write(text)

    # return mod
