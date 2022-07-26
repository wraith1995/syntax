""" A module for parsing ASDL grammars into Python Class hierarchies
    Adopted from from https://raw.githubusercontent.com/gilbo/atl/master/ATL/adt.py

"""

import asdl
from types import ModuleType
from typing import Callable, Optional
from collections.abc import Sequence, Mapping
from weakref import WeakValueDictionary
from dataclasses import make_dataclass, field
from ilist import ilist
import abc

class ADTCreationError(Exception):
    pass

class GenericADTError(Exception):
    pass

def _asdl_parse(str):
    parser = asdl.ASDLParser()
    module = parser.parse(str)
    return module

def _build_superclasses(asdl_mod):
    scs = {}
    isprod = {}
    def create_invalid_init(nm):
        def invalid_init(self):
            raise GenericADTError(f"{nm} should never be instantiated")
        return invalid_init
    
    for nm, v in asdl_mod.types.items():
        if isinstance(v, asdl.Sum):
            scs[nm] = type(nm, (), {"__init__" : create_invalid_init(nm)})
            isprod[nm] = False
        elif isinstance(v, asdl.Product):
            scs[nm] = type("_" + nm, (), {"__init__" : create_invalid_init(nm)})
            isprod[nm] = True
    return scs, isprod

_builtin_checks = {
    'object'  : lambda x: x is not None
}

_builtin_types = {
    'string' : str,
    'str' : str,
    'int' : int,
    'object' : object,
    'float' : float,
    'bool' : bool
}

def _build_checks(asdl_mod, scs, ext_checks):
    checks = _builtin_checks.copy()
    def make_check(sc):
        return lambda x: isinstance(x,sc)
    
    for nm in ext_checks:
        checks[nm] = ext_checks[nm]
    for nm in scs: # I think this is unneeded.
        if nm not in checks:
            raise ADTCreationError(f"Name conflict for type '{nm}'")
        sc = scs[nm]
        checks[nm] = make_check(sc)
    return checks

def _build_types(SC, ext_types):
    tys = _builtin_types.copy()
    for x in ext_types.keys():
        tys[x] = ext_types[x]
    for x in SC.keys(): # CHECK ME: Is overwriting the right choice here?
        tys[x] = SC[x]
    return tys
        

#Factorize: Get mod to call
#Factorize: element type maker - first just try - then if it is not iterable

def build_field_data(cname, field_spec, CHK, TYS):
    fields = [] #FIXME: These should all be nametuple for readability.
    field_data = []
    chks = []
    for f in field_spec:
        chk = lambda x: True
        name  = f.name
        if name is None:
            name = str(f.type) #FIXME: I am not sure this is correct
        seq = f.seq
        opt = f.opt
        tys = TYS[str(f.type)]
        if str(tys) in CHK:
            chk = CHK[str(tys)]
        field_data.append([seq, opt, f.type])
        chks.append(chk)
        # Exact resolution of these options is unclear to me.
        if opt and not seq: 
            if tys in defaults:
                default = defaults[tys]
                if isinstance(default, tys):
                    fd = (name, tys, field(default=default))
                elif isinstance(default, Callable):
                    fd = (name, tys, field(default_factory=default))
                else:
                    raise ADTCreationError("Default contains a type that is not correct or is not a callable.")
            else:
                fd = (name, Optional[tys], None)
        elif not opt and seq: # should this be a non-empty list?
            fd = (name, ilist[tys])
        elif opt and seq:
            fd = (name, ilist[tys], ilist([]))
        else:
            fd = (name, tys)
        fields.append(fd)
    return (fields, field_data, chks)
    

def build_dc(cname, field_info, fieldData, ISPROD, constructorDict,  Err,
             parent=None, memoize=True, namespace_injector=None, defaults={}):
    if parent is not None:
        bases = (parent,)
    else:
        bases = tuple()
        raise ADTCreationError("Creating a dataclass that supports weakrefs requires either slots=False or a parent class that supports weakref; we only support the latter.")
    (fields, field_data, chks) = field_info
    classdict = WeakValueDictionary({})
    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        cls.__init__(obj, *args, **kwargs) # build the data class to check if it exists. Hope this is gc'd quickly.
        if (obj) in classdict:
            return classdict[(obj)]
        else:
            classdict[(obj)] = obj
            return obj
    def element_checker(cname, fieldName, targetType, tyname, chk, opt, x):
        xt = type(x)
        badType = Err("{0}.{1} does not have type {2} because a value has type {3}".format(cname, fieldName, targetType, xt))
        badCheck = Err("{0}.{1} is not valid because {2} failed the check for type {3}".format(cname, fieldName, x, etype))
        badSeq = Err("{0}.{1} does not have type {2}, and instead has type {3};
                       we tried to convert the value, {4}, because it was a sequence or mapping, but this failed.".format(cname, fieldName, targetType, xt, x))
        badElem = Err("{0}.{1} does not have type {2}, and instead has type {3};
                       we tried to convert the value, {4}, because it was the single correct type, but this failed.".format(cname, fieldName, targetType, xt, x))

        #badTypeConstruction = None #three cases: it is map - it is a sequence - with the right number of required args.

        earlyAble = tyname in ISPROD and ISPROD[tyname]
        minArgs = 0
        maxArgs = 0
        singleType = None
        if earlyAble:
            (ofields, ofield_data, _) = fieldData[tyname]
            maxArgs = len(ofield_data)
            minArgs = len(filter(lambda x: not x[1], ofield_data)) #not optional fields
            if minArgs == 1:
                loc = -1
                for (idx, x) in enumerate(ofield_data):
                    if not x[1]:
                        loc = idx
                        break
                singleType = ofields[loc][1]
                        
        convert = False
        if not isinstance(x, targetType):
            if not earlyAble:
                raise badType
            else:
                if isinstance(x, Sequence):
                    if minArgs <= len(x) <= maxArgs:
                        try:
                            x = constructorDict[tyname](*x)
                            convert = True
                        except:
                            raise badSeq
                elif isinstance(x, mapping):
                    if minArgs <= len(x) <= maxArgs:
                        try:
                            x = constructorDict[tyname](**x)
                            convert = True
                        except:
                            raise badSeq
                elif singleType is not None and isinstance(x, singleType): #How does opt interact with this case?
                    try:
                        x = constructorDict[tyname](x)
                        convert = True
                    except:
                        raise badElem
                else:
                    raise badType
        else:
            pass

        if not (x is None and opt) and not chk(x):
            raise badCheck
        else:
            return (convert, x)


    def __post_init__(self):
        for (fd, fds, chk) in zip(fields, field_data, chks):
            (seq, opt, tyname) = fds
            fieldName = fd[0]
            typeName = fds[2]
            opt = fds[1]
            val = getattr(self, fd[0])
            actual_type = type(val)
            expected_type = fd[1]
            # Check the sequence-ness 
            if seq:
                if isinstance(val, abc.Iterable):
                    val = ilist(val)
                    #setattr(self, fd[0], val)
                else:
                    raise Err("{0}.{1} must be an iterable, but it has type {2}".format(cname, fd[0], str(actual_type)))
                # check the value of each element:
                etype = fd[1].__args__[0]
                vals = []
                for x in val:
                    (_, xp) = element_checker(cname, fieldName, etype, typeName, chk, opt, x)
                    vals.append(xp)
                valsp = ilist(vals)
                setattr(self, fieldName, valsp)
                    # if not isinstance(x, etype):
                    #     xt = type(x)
                    #     raise Err("{0}.{1} does not have type {2} because a value has type {3}".format(cname, fd[0], fd[1], xt))
                    # elif not chk(val):
                    #     raise Err("{0}.{1} is not valid because {2} failed the check for type {3}".format(cname, fd[0], x, etype))
                    # else:
                    #     pass
            else:
                (convert, xp) = element_checker(cname, fieldName, etype, typeName, chk, opt, x)
                if convert:
                    setattr(self, fieldName, xp)
                # if isinstance(val, fd[1]):
                #     if not chk(val) and not (val is None and opt):
                #         raise Err("{0}.{1} is not valid because {2} failed the check for type {3}".format(cname, fd[0], val, fd[1]))
                #     else:
                #         pass
                # else:
                #     xt = type(val)
                #     raise Err("{0}.{1} has type {2}, but should have type {3}".format(cname, fd[0], xt, fd[1]))
    namespace = {}
    if namespace_injector is not None:
        namespace = namespace_injector(cname, fields, field_data, parent)
    if memoize:
        namespace["__post_init__"] = __post_init__
        namespace["__new__"] = __new__
    else:
        namespace["__post_init__"] = __post_init__

    return make_dataclass(cname, fields, bases=bases, frozen=True, slots=True, namespace=namespace)

def _build_classes(asdl_mod, ext_checks={},
                   ext_types={}, memoize=True, namespace_injector=None, defaults={}):
    SC, ISPROD   = _build_superclasses(asdl_mod)
    CHK  = _build_checks(asdl_mod, SC, ext_checks)
    TYS = _build_types(SC, ext_types)
    mod  = ModuleType(asdl_mod.name)
    Err  = type(asdl_mod.name + " Error", (Exception,), {})
    constructorDict = {}
    fieldData = {}
    for nm, t in asdl_mod.types.items():
        match t:
            case asdl.Product:
                fds = build_field_data(nm, t.fields, CHK, TYS)
                fieldData[nm] = fds
            case asdl.Sum:
                for c in t.types:
                    fds = build_field_data(c.name, c.fields + t.attributes, CHK, TYS)
                    fieldData[(nm, c.name)] = fds
            case _:
                raise ADTCreationError("Unexpected kind of asdl type: neither Sum nor Product.")
                    
    
    def create_prod(nm, t, T):
        C = build_dc(nm, fieldData[nm], fieldData, ISPROD, constructorDict,
                     Err, parent=T, memoize=memoize,
                     namespace_injector=namespace_injector, defaults=defaults)
        constructorDict[nm] = C
        return C
    
    def create_sum_constructor(tname, cname, T):
        C = build_dc(cname, fieldData[(tname, cname)], fieldData, ISPROD, constructorDict,
                     Err, parent=T, memoize=memoize,
                     namespace_injector=namespace_injector, defaults=defaults)
        constructorDict[(tname, cname)] = C
        return C

    def create_sum(typ_name, t):
        T          = SC[typ_name]
        afields    = t.attributes
        for c in t.types:
            C      = create_sum_constructor(typ_name, c.name, T)
            if not hasattr(mod, c.name):
                raise ADTCreationError(f"name '{c.name}' conflict in module '{mod}'")
            setattr(T, c.name, C)
            setattr(mod, c.name, C)
        return T
    
    for nm,t in asdl_mod.types.items():
        if isinstance(t, asdl.Product):
            setattr(mod, nm, create_prod(nm, t, SC[nm]))
        elif isinstance(t, asdl.Sum):
            setattr(mod, nm, create_sum(nm, t))
        else:
            raise ADTCreationError("Unexpected kind of asdl type: neither Sum nor Product.")

    return mod

def ADT(asdl_str, ext_types={}, ext_checks={}, defaults={}, memoize=True):
    """ Function that converts an ASDL grammar into a Python Module.

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
    mod      = _build_classes(asdl_ast, ext_checks=ext_checks, ext_types=ext_types, memoize=memoize, defaults=defaults)
    # cache values in case we might want them
    mod._ext_checks = ext_checks
    mod._ext_types = ext_types
    mod._ast        = asdl_ast
    mod._defstr     = asdl_str
    mod._defaults = defaults
    mod._memoize = memoize

    mod.__doc__     = (f"ASDL Module Generated by ADT\n\n"
                       f"Original ASDL description:\n{asdl_str}")
    return mod


