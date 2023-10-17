"""A module for parsing ASDL grammars into Python Class hierarchies.

Adopted from https://raw.githubusercontent.com/gilbo/atl/master/ATL/adt.py
And again from https://github.com/ChezJrk/asdl/blob/master/src/asdl_adt/adt.py.
"""
import inspect  # noqa: F401
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Collection, Iterable, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import Field, field, make_dataclass, replace
from itertools import chain
from types import ModuleType
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

defaultsTy = Mapping[Union[str, type, Tuple[str, str], Tuple[str, type]], Any]


indent = "    "


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
    ty: type
    chk: Callable
    hasDefault: bool
    default: Any


def fdtypestr(fd: field_data, env) -> str:
    """Represent a field type as a string."""
    tyname = fd.ty.__name__
    if env.isInterallyDefined(fd.ty):
        tyname += "_type"
    if fd.seq:
        return "typing.Sequence[{0}]".format(tyname)
    elif fd.opt:
        return "typing.Optional[{0}]".format(tyname)
    else:
        return tyname


def fdinitstr(fd: field_data, env) -> str:
    """Represent type signature of an init."""
    tystr = fdtypestr(fd, env)
    start = "{0}:{1}".format(fd.name, tystr)
    if fd.hasDefault:
        start += " = ..."
    return start


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
        ty,
        checks[str(f.type)] if str(f.type) in checks else dumb,
        hasDefault,
        default,
    )


class constructor_data(NamedTuple):
    """Represent a constructor."""

    sup: type
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
            self.constructorData[name] = constructor_data(ty, fieldData, name, maxArgs, minArgs, minSet)
        self.allTypeNames = set().union(*[set(v) for v in self.typeCollections.values()])

    def isInterallyDefined(self, typ: type) -> bool:
        """Determine if a type is internally defined."""
        return issubclass(typ, self.sumClass) or issubclass(typ, self.prodClass)

    def isInternalSum(self, typ: type) -> bool:
        """Determine if a type is an internally defined sum type."""
        return issubclass(typ, self.sumClass)

    def isInternalProduct(self, typ: type) -> bool:
        """Determine if a type is an internally defined product type."""
        return issubclass(typ, self.prodClass)

    def generateClassStub(self, name: str) -> List[str]:
        """Generate stub file snippets for a class."""
        data = []
        cdata = self.constructorData[name]
        constantCheck = len(cdata.fields) == 0
        if issubclass(cdata.sup, self.sumClass):
            superName = "_" + cdata.sup.__name__
        else:
            superName = "object"

        data.append("class {0}({1}):".format(name if not constantCheck else "_" + name, superName))
        for fd in cdata.fields:
            tystr = fdtypestr(fd, self)
            data.append(indent + "{0}: {1}".format(fd.name, tystr))
        data.append(
            indent
            + "__match_args__ = ({0})".format(", ".join(['"' + fd.name + '"' for fd in cdata.fields]))  # noqa: W503
        )
        inits = ["self"] + [fdinitstr(fd, self) for fd in cdata.fields]
        initstr = ", ".join(inits)
        data.append(indent + "def __init__({0}) -> None: ...".format(initstr))
        if len(cdata.fields) == 0:
            data.append("\n")
            data.append("{0}: _{0} = _{0}()".format(name))
            data.append("\n")
        if self.options.loop:
            data.append(indent + "def loop(self, internal: bool = True) -> typing.Iterator[Any]: ...\n")
        return data

    def generateStub(self, oname: str) -> str:
        """Generate stub file for an ADT."""
        stub_commands = [
            "import abc",
            "import typing",
            "from syntax import stamp",
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


def build_post_init(fieldData, Err, cname, element_checker):
    """Build dataclass post init function."""

    def __post_init__(self):
        for fd in fieldData:
            seq = fd.seq
            opt = fd.opt
            chk = fd.chk
            fieldName = fd.name
            ty = fd.ty
            val = getattr(self, fieldName)
            actual_type = type(val)
            # Check the sequence-ness
            if seq:
                if isinstance(val, Iterable):
                    val = tuple(val)
                elif PyVar is not None and isinstance(val, PyVar):
                    return None
                else:
                    raise Err(
                        """{0}.{1} must be iterable,
                    but it has type {2}""".format(
                            cname, fieldName, actual_type
                        )
                    )
                # check the value of each element:
                vals = []
                for x in val:
                    (_, xp) = element_checker(cname, fieldName, ty, chk, opt, x)
                    vals.append(xp)
                valsp = tuple(vals)
                object.__setattr__(self, fieldName, valsp)
                # https://github.com/python/cpython/blob/bceb197947bbaebb11e01195bdce4f240fdf9332/Lib/dataclasses.py#L565
                # Validity of this strategy is based on a careful reading
                # of the dataclass implementation. In particular:
                # 1. _post_init is called in init
                # 2. hash is not precomputed before _post_init or cached
                # 3. frozen is just a promise
                # Ergo, when we init an object for caching, our hash
                # is correct even with all of this frozen breaking
                # nonsense in _post_init.
            else:
                (convert, xp) = element_checker(cname, fieldName, ty, chk, opt, val)
                if convert:
                    object.__setattr__(self, fieldName, xp)  # GOD I AM SORRY.

    return __post_init__


def build_dc(
    env: ADTEnv,
    cname: str,
    parent: type,
    fieldData: list[field_data],
    Err: type,
    mod: ModuleType,
    memoize: bool = True,
    namespace_injector=None,
    visitor: bool = True,
    slots: bool = True,
):
    """Build a dataclass for an ADT type."""
    isConstant = len(fieldData) == 0
    classdict: WeakValueDictionary = WeakValueDictionary({})

    egraphIsInstance = build_egraph_instance_check(env)

    __new__ = build_new(memoize, classdict)

    element_checker = build_element_check(mod, egraphIsInstance, Err, env, cname)
    __post_init__ = build_post_init(fieldData, Err, cname, element_checker)

    __iter__, _map = build_element_iteration_methods(Err, fieldData, env)

    mcopy, update, dcopy = build_element_copy_methods(fieldData, env)

    isdisjoint, isomorphism, __isomorphism__ = build_function_category_methods(fieldData, env, Err)

    namespace = {}
    if namespace_injector is not None:
        namespace = namespace_injector(cname, parent, fieldData, Err, parent)
    if memoize:
        namespace["__post_init__"] = __post_init__
        namespace["__new__"] = __new__
    else:
        namespace["__post_init__"] = __post_init__
    if env.options.copyMethods:
        namespace["__copy__"] = mcopy
        namespace["copy"] = mcopy
        namespace["update"] = update
        namespace["__deepcopy__"] = dcopy
    # namespace["__contains__"] = __contains__
    if env.options.setfunctions:
        namespace["isdisjoint"] = isdisjoint
        namespace["isomorphism"] = isomorphism
        namespace["__isomorphism__"] = __isomorphism__
    if env.options.loop:
        namespace["loop"] = __iter__
    if env.options.mapper:
        namespace["map"] = _map
    if isConstant:
        namespace["__call__"] = lambda self: self
    if env.anyEgraph():
        # namespace["__match_args__"] = __match_args__
        pass
    if visitor or env.options.visit:
        accept = build_visitor_accept(fieldData)

        namespace["accept"] = accept

    def fieldp(x) -> Field:
        tmp = field(default_factory=x) if callable(x) else field(default=x)
        assert isinstance(tmp, Field)
        return tmp

    bf = field(default=("___" + cname + "__"), init=False, repr=False)
    assert isinstance(bf, Field)
    extra: List[Tuple[str, Type[Any], Field]] = [("___" + cname + "__", str, bf)]
    fields: List[Union[Tuple[str, Type[Any], Field], Tuple[str, Type[Any]]]] = [
        (fd.name, fd.ty) if not fd.hasDefault else (fd.name, fd.ty, fieldp(fd.default)) for fd in fieldData
    ]
    fields += extra
    try:
        cls = make_dataclass(
            cname,
            fields,
            bases=(parent,),
            frozen=True,
            slots=slots,
            namespace=namespace,
        )
    except BaseException:
        raise ADTCreationError("Failed to crate class for {0} with fields {1}".format(cname, fields))
    if isConstant:
        val = cls()
        return (val, cls)
    else:
        return cls


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


def build_element_check(mod, egraphIsInstance, Err, env, cname):
    """Add method to check arguments and convert as needed."""

    def element_checker(
        cname: str,
        fieldName: str,
        targetType: type,
        chk: Callable,
        opt: bool,
        x: Any,
    ):
        badType, badSeq, badElem, badCheck = build_local_adt_errors(Err, x, cname, fieldName, targetType, opt)

        earlyAble = env.isInternalProduct(targetType)
        tyname = targetType.__name__
        singleType: Optional[type] = None
        if earlyAble:
            minArgs = env.constructorData[tyname].minArgs
            maxArgs = env.constructorData[tyname].maxArgs
            if minArgs == 1:
                sf = env.constructorData[tyname].minSatisfy[0]
                singleType = sf.ty  # list[sf.ty] if sf.seq else sf.ty
        else:
            minArgs = sys.maxsize
            maxArgs = sys.maxsize
        convert = False
        if x is None and opt:
            return (False, None)
        if not egraphIsInstance(x, targetType):
            if not earlyAble:
                raise badType
            else:
                if isinstance(x, Sequence):
                    if minArgs <= len(x) <= maxArgs:
                        try:
                            x = getattr(mod, tyname)(*x)
                            convert = True
                        except BaseException:
                            raise badSeq
                elif isinstance(x, Mapping):
                    if minArgs <= len(x) <= maxArgs:
                        try:
                            x = getattr(mod, tyname)(**x)
                            convert = True
                        except BaseException:
                            raise badSeq
                elif singleType is not None and egraphIsInstance(x, singleType):
                    # CHECK: How does opt interact with this case?
                    # CHECK: How does this interact with seq case?
                    try:
                        x = getattr(mod, tyname)(x)
                        convert = True
                    except BaseException:
                        raise badElem
                elif x is None and opt:
                    pass
                else:
                    raise badType
        else:
            pass

        if not (x is None and opt) and not chk(x):
            raise badCheck
        else:
            return (convert, x)

    return element_checker


def build_egraph_instance_check(env):
    """Add methods to manage use of egraphs."""

    def egraphIsInstance(val, ty):
        if isinstance(val, ty):
            return True
        else:
            tyName = ty.__name__
            if env.useEgraph(tyName):
                return (PyVar is not None and isinstance(val, PyVar)) or isinstance(val, str)

    return egraphIsInstance


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
    if memoize is True:
        memoize = set(env.constructorData.keys())
    elif memoize is False:
        memoize = set()
    elif isinstance(memoize, set):
        pass
    else:
        raise ADTCreationError("Memoization should be a set or Bool")

    mod = _build_classes(asdl_ast, env, memoize=memoize, slots=slots, visitor=visitor)
    # cache values in case we might want them
    setattr(mod, "_ast", asdl_ast)  # noqa: B010
    # mod._ast = asdl_ast
    setattr(mod, "_defstr", asdl_str)  # noqa: B010
    # mod._defstr = asdl_str
    setattr(mod, "_env", env)
    mod.__doc__ = f"""ASDL Module Generated by ADT\n\n"
    f"Original ASDL description:\n{asdl_str}"""
    if stubfile is not None:
        with open(stubfile, "w+") as f:
            text = env.generateStub("")
            f.write(text)

    return mod
