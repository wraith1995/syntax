""" A module for parsing ASDL grammars into Python Class hierarchies
    Adopted from https://raw.githubusercontent.com/gilbo/atl/master/ATL/adt.py
    And again from https://github.com/ChezJrk/asdl/blob/master/src/asdl_adt/adt.py
"""

import asdl
from types import ModuleType
from typing import Callable, Any, Union, NamedTuple, Optional
from collections.abc import Sequence, Mapping, Collection, Iterable
from collections import OrderedDict
from weakref import WeakValueDictionary
from dataclasses import make_dataclass, field, replace
from copy import copy, deepcopy
from abc import ABC, abstractmethod
from itertools import chain


class ADTCreationError(Exception):
    pass


class GenericADTError(Exception):
    pass


class _AsdlAdtBase(ABC):
    @abstractmethod
    def __init__(self):
        assert False, "Should be unreachable."


class _ProdBase(_AsdlAdtBase):
    pass


class _SumBase(_AsdlAdtBase):
    pass


def _asdl_parse(str):
    parser = asdl.ASDLParser()
    module = parser.parse(str)
    return module


_builtin_checks = {
    'object': lambda x: x is not None
}

_builtin_types = {
    'string': str,
    'str': str,
    'int': int,
    'object': object,
    'float': float,
    'bool': bool
}


class field_data(NamedTuple):
    name: str
    seq: bool
    opt: bool
    ty: type
    chk: Callable
    hasDefault: bool
    default: Any


def build_field_cdata(outerName, f, externalTypes,
                      internalTypes, checks, defaults):
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
    return field_data(f.name, f.seq, f.opt,
                      ty,
                      checks[str(f.type)] if str(f.type) in checks else lambda _: True,
                      hasDefault, default)


class constructor_data(NamedTuple):
    sup: type
    fields: list[field_data]
    name: str
    minArgs: int
    maxArgs: int
    minSatisfy: list[field_data]


class ADTEnv:
    def __init__(self, name, adsl_adt, external_types, defaults, checks):
        self.name = name
        self.checks = checks
        self.defaults = defaults  # (name, field), (name, type), type
        self.sumClass = type("sum_" + name, (_SumBase,), {})
        self.prodClass = type("prod_" + name, (_ProdBase,), {})
        self.superTypes = dict()
        self.externalTypes = external_types
        self.constructorData = dict()

        def fieldValidator(flds, name, names=set()):
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
                myattrs = set()
                fieldValidator(ty.fields, name, names=myattrs)
                self.constructorData[name] = (ty.fields, typ)
            elif isinstance(ty, asdl.Sum):
                typ = type(name, (self.sumClass,), {})
                self.superTypes[name] = typ
                myattrs = set()
                fieldValidator(ty.attributes, name, names=myattrs)
                for summand in ty.types:
                    if summand.name in self.constructorData:
                        raise ADTCreationError("{0} conflicts with another name already defined".format(summand.name))
                    fieldValidator(summand.fields, summand.name, names=myattrs.copy())
                    self.constructorData[summand.name] = ((summand.fields + ty.attributes, typ))
            else:
                raise ADTCreationError("ASDL item not sum nor product.")
        for (name, (fields, ty)) in self.constructorData.copy().items():
            fieldData = [build_field_cdata(name, f, self.externalTypes,
                                           self.superTypes,
                                           self.checks,
                                           self.defaults)
                         for f in fields]
            maxArgs = len(fieldData)
            minSet = list(filter(lambda x: not x.hasDefault, fieldData))
            minArgs = len(minSet)
            self.constructorData[name] = constructor_data(ty, fieldData, name,
                                                          maxArgs, minArgs,
                                                          minSet)

    def isInterallyDefined(self, typ):
        return issubclass(typ, self.sumClass) or issubclass(typ, self.prodClass)

    def isInternalSum(self, typ):
        return issubclass(typ, self.sumClass)

    def isInternalProduct(self, typ):
        return issubclass(typ, self.prodClass)


def build_dc(env, cname, parent, fieldData, Err, mod,
             memoize=True,
             namespace_injector=None,
             slots=True):
    classdict = WeakValueDictionary({})

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        cls.__init__(obj, *args, **kwargs)
        # build the data class to check if it exists.
        # Hope this is gc'd quickly.
        if obj in classdict:
            return classdict[(obj)]
        else:
            classdict[obj] = obj
            return obj

    def element_checker(cname, fieldName, targetType, chk, opt, x):
        xt = type(x)
        badType = Err("""{0}.{1} does not have type
        {2} because a value {3}
        has type {4} and opt={5}""".format(cname, fieldName, targetType, x, xt, opt))
        badCheck = Err("""{0}.{1} is not valid because
        {2} failed the
        check for type {3}""".format(cname, fieldName, x, targetType))
        badSeq = Err("""{0}.{1} does not have type {2},
        and instead has type {3};
        we tried to convert the value, {4}, because it was a
        sequence or mapping,
        but this failed.""".format(cname, fieldName, targetType, xt, x))
        badElem = Err("""{0}.{1} does not have type {2},
        and instead has type {3};
        we tried to convert the value, {4}, because it was the single
        correct type,
        but this failed.""".format(cname, fieldName, targetType, xt, x))

        earlyAble = env.isInternalProduct(targetType)
        tyname = targetType.__name__
        singleType: Optional[type] = None
        if earlyAble:
            minArgs = env.constructorData[tyname].minArgs
            maxArgs = env.constructorData[tyname].maxArgs
            if minArgs == 1:
                sf = env.constructorData[tyname].minSatisfy[0]
                singleType = sf.ty #list[sf.ty] if sf.seq else sf.ty
        else:
            minArgs = int("infty")
            maxArgs = int("infty")
        convert = False
        if x is None and opt:
            return (False, None)
        if not isinstance(x, targetType):
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
                elif singleType is not None and isinstance(x, singleType):
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
                else:
                    raise Err("""{0}.{1} must be iterable,
                    but it has type {2}""".format(cname,
                                                  fieldName,
                                                  actual_type))
                # check the value of each element:
                vals = []
                for x in val:
                    (_, xp) = element_checker(cname, fieldName,
                                              ty, chk, opt, x)
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
                (convert, xp) = element_checker(cname, fieldName,
                                                ty, chk, opt, val)
                if convert:
                    object.__setattr__(self, fieldName, xp)  # GOD I AM SORRY.


    def map(self, mapper):
        if self in mapper:
            return mapper[self]
        rep = {}
        for fd in fieldData:
            test = env.isInterallyDefined(fd.ty)
            temp = getattr(self, fd.name)
            iters = []
            if fd.seq:
                if temp in mapper:
                    rep[fd.name] = mapper[temp]
                    continue
                else:
                    iters = temp
            reps = [mapper[x] if x in mapper else (x.map(mapper) if test else x)
                   for x in iters]
            if not fd.seq:
                rep[fd.name] = reps[0]
            else:
                rep[fd.name] = reps
        return replace(self, **rep)


    def mcopy(self, deep=False, copies={}, ignore=set(), onlyCopies=False,
              copyInternal=True, copyExternal=False):
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
                            item_c = item.__copy__(deep=deep,
                                                   copies=copies,
                                                   ignore=ignore,
                                                   onlyCopies=onlyCopies,
                                                   copyInternal=copyInternal,
                                                   copyExternal=copyExternal)
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
                    for l in lst:
                        res.append(deepcopy(l) if deep else copy(l))
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

    def __contains__(self, other):
        if self == other:
            return True
        else:
            for fd in fieldData:
                temp = getattr(self, fd.ty)
                if env.isInterallyDefined(fd.ty):
                    if temp.__contains__(other):
                        return True
                    else:
                        pass
            return False

    def __iter__(self, internal: bool = True):
        yield self
        # go over fields that have interally defined types and iterate though
        # if they are lists, iterate through them tooo
        # make sure nothing is None.
        nexts = chain(*[getattr(self, fd.name)
                        if not fd.seq
                        else chain(*(getattr(self, fd.name)))
                        for fd in fieldData
                        if (not internal or env.isInterallyDefined(fd.ty))
                        and getattr(self, fd.name) is not None
                        ])
        yield from nexts
    # FIXME: We clearly need types of iterations for this.
    # Iterate over the internal definitions vs over the external definitions.
    # FIXME: I should not be using fields here I think? Use dataclasses's

    def isdisjoint(self, other):
        if self in other:
            return False
        else:
            for fd in fieldData:
                temp = getattr(self, fd.fieldName)
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
    namespace = {}
    if namespace_injector is not None:
        namespace = namespace_injector(cname, parent, fieldData, Err, parent)
    if memoize:
        namespace["__post_init__"] = __post_init__
        namespace["__new__"] = __new__
    else:
        namespace["__post_init__"] = __post_init__
    namespace["__copy__"] = mcopy
    namespace["copy"] = mcopy
    namespace["update"] = update
    namespace["__deepcopy__"] = dcopy
    # namespace["__contains__"] = __contains__
    namespace["isdisjoint"] = isdisjoint
    namespace["isomorphism"] = isomorphism
    namespace["__isomorphism__"] = __isomorphism__
    namespace["loop"] = __iter__
    namespace["map"] = map

    def fieldp(x):
        return field(default_factory=x) if isinstance(x, Callable) else field(default=x)
    fields = [(fd.name, fd.ty) if not fd.hasDefault else
              (fd.name, fd.ty, fieldp(fd.default))
              for fd in fieldData] + [("___" + cname + "__", str,
                                       field(default=("___" + cname + "__"),
                                             init=False,
                                             repr=False))]
    try:
        cls = make_dataclass(cname, fields, bases=(parent,),
                             frozen=True,
                             slots=slots, namespace=namespace)
    except Exception as _:
        raise Exception("Failed to crate class for {0} with fields {1}".format(cname, fields))
    return cls


def _build_classes(asdl_mod, env, memoize,
                   namespace_injector=None, slots=False):
    mod = ModuleType(asdl_mod.name)
    Err: Exception = type(asdl_mod.name + "Error", (Exception,), {})
    setattr(mod, "__err__", Err)
    for (name, ty) in env.superTypes.items():
        setattr(mod, name, ty)
    for (name, data) in env.constructorData.items():
        dc = build_dc(env, data.name, data.sup, data.fields, Err, mod,
                      memoize=name in memoize,
                      namespace_injector=namespace_injector,
                      slots=slots)
        setattr(mod, name, dc)
    return mod


def ADT(asdl_str: str,
        ext_types: Mapping[str, Callable] = {},
        ext_checks: Mapping[str, type] = {},
        defaults: Mapping[str, Any] = {},
        memoize: Union[Collection[str], bool] = True,
        slots: bool = False):
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
    assert isinstance(asdl_ast, asdl.Module)

    env = ADTEnv(asdl_ast.name, asdl_ast,
                 _builtin_types | ext_types,
                 defaults,
                 _builtin_checks | ext_checks)
    if memoize is True:
        memoize = set(env.constructorData.keys())
    elif memoize is False:
        memoize = set()
    elif isinstance(memoize, set):
        pass
    else:
        raise ADTCreationError("Memoization should be a set or Bool")

    mod = _build_classes(asdl_ast, env, memoize=memoize, slots=slots)
    # cache values in case we might want them
    setattr(mod, "_ast", asdl_ast)
    # mod._ast = asdl_ast
    setattr(mod, "_defstr", asdl_str)
    #mod._defstr = asdl_str
    setattr(mod, "_env", env)
    mod.__doc__ = (f"ASDL Module Generated by ADT\n\n"
                   f"Original ASDL description:\n{asdl_str}")
    return mod
