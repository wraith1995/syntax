""" A module for parsing ASDL grammars into Python Class hierarchies
    Adopted from https://raw.githubusercontent.com/gilbo/atl/master/ATL/adt.py

"""

import asdl
from types import ModuleType
from typing import Callable, Optional
from collections.abc import Sequence, Mapping
from collections import OrderedDict
from weakref import WeakValueDictionary
from dataclasses import make_dataclass, field, is_dataclass, replace
from copy import copy, deepcopy
from .ilist import ilist
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
            scs[nm] = type(nm, (),
                           {"__init__": create_invalid_init(nm)})
            isprod[nm] = False
        elif isinstance(v, asdl.Product):
            scs[nm] = type("_" + nm, (),
                           {"__init__": create_invalid_init(nm)})
            isprod[nm] = True
    return scs, isprod


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


def _build_checks(asdl_mod, scs, ext_checks):
    checks = _builtin_checks.copy()

    def make_check(sc):
        return lambda x: isinstance(x, sc)

    for nm in ext_checks:
        checks[nm] = ext_checks[nm]
    for nm in scs:
        # I think this is unneeded.
        if nm in checks:
            raise ADTCreationError(f"Name conflict for type '{nm}'")
        sc = scs[nm]
        checks[nm] = make_check(sc)
    return checks


def _build_types(SC, ext_types):
    tys = _builtin_types.copy()
    for x in ext_types.keys():
        tys[x] = ext_types[x]
    for x in SC.keys():
        # CHECK ME: Is overwriting the right choice here?
        tys[x] = SC[x]
    return tys


def build_field_data(cname, field_spec, CHK, TYS, defaults):
    fields = []  # FIXME: These should all be nametuple for readability.
    field_data = []
    chks = []
    for f in field_spec:
        chk = lambda x: True
        name = f.name
        if name is None:
            name = str(f.type)  # FIXME: I am not sure this is correct
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
                    raise ADTCreationError("""Default contains a type that
                    is not correct or is not a callable.""")
            else:
                fd = (name, Optional[tys], None)
        elif not opt and seq:  # should this be a non-empty list?
            fd = (name, ilist[tys])
        elif opt and seq:
            fd = (name, ilist[tys], ilist([]))
        else:
            fd = (name, tys)
        fields.append(fd)
    return (fields, field_data, chks)


def build_dc(cname, field_info, fieldData, ISPROD, constructorDict,
             internallyDefined, Err, parent=None, memoize=True,
             namespace_injector=None, defaults={}):
    if parent is not None:
        bases = (parent,)
    else:
        bases = tuple()
        raise ADTCreationError("""Creating a dataclass that supports weakrefs requires either slots=False
        or a parent class that supports weakref;
        we only support the latter.""")
    (fields, field_data, chks) = field_info
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

    def element_checker(cname, fieldName, targetType, tyname, chk, opt, x):
        xt = type(x)
        badType = Err("""{0}.{1} does not have type
        {2} because a value
        has type {3}""".format(cname, fieldName, targetType, xt))
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

        earlyAble = tyname in ISPROD and ISPROD[tyname]
        minArgs = 0
        maxArgs = 0
        singleType = None
        if earlyAble:
            (ofields, ofield_data, _) = fieldData[tyname]
            maxArgs = len(ofield_data)
            minArgs = len(list(filter(lambda x: not x[1], ofield_data)))
            if minArgs == 1:
                loc = -1
                for (idx, y) in enumerate(ofield_data):
                    if not y[1]:
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
                elif isinstance(x, Mapping):
                    if minArgs <= len(x) <= maxArgs:
                        try:
                            x = constructorDict[tyname](**x)
                            convert = True
                        except:
                            raise badSeq
                elif singleType is not None and isinstance(x, singleType):
                    # How does opt interact with this case?
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
            # Check the sequence-ness
            if seq:
                if isinstance(val, abc.Iterable):
                    val = ilist(val)
                else:
                    raise Err("""{0}.{1} must be iterable,
                    but it has type {2}""".format(cname,
                                                  fd[0],
                                                  str(actual_type)))
                # check the value of each element:
                etype = fd[1].__args__[0]
                vals = []
                for x in val:
                    (_, xp) = element_checker(cname, fieldName,
                                              etype, typeName, chk, opt, x)
                    vals.append(xp)
                valsp = ilist(vals)
                object.__setattr__(self, fieldName, valsp)
            else:
                (convert, xp) = element_checker(cname, fieldName,
                                                fd[1], typeName, chk, opt, val)
                if convert:
                    object.__setattr__(self, fieldName, xp)  # GOD I AM SORRY.

    def mcopy(self, deep=False, copies={}, complete=False):
        d = {}
        for fd in fields:
            temp = getattr(self, fd[0])
            if temp in copies and not complete:
                d[fd[0]] = copies[temp]
            elif type(temp) in internallyDefined:  # Internal definitions.
                d[fd[0]] = temp.__copy__(deep=deep,
                                         copies=copies,
                                         complete=complete)
                copies[temp] = d[fd[0]]
            else:
                d[fd[0]] = deepcopy(temp) if deep else copy(temp)

        return replace(self, **d)

    def dcopy(self):
        return mcopy(self, deep=True)

    def __contains__(self, other):
        if self == other:
            return True
        else:
            for fd in fields:
                temp = getattr(self, fd[0])
                if type(temp) in internallyDefined:
                    if temp.__contains__(other):
                        return True
                    else:
                        pass
            return False

    # FIXME: We clearly need types of iterations for this.
    # Iterate over the internal definitions vs over the external definitions.
    # FIXME: I should not be using fields here I think? Use dataclasses's
    def isdisjoint(self, other):
        if self in other:
            return False
        else:
            for fd in fields:
                temp = getattr(self, fd[0])
                if type(temp) in internallyDefined:
                    if temp.isdisjoint(other):
                        pass
                    else:
                        return False
            return True

    def __isomorphism__(x, y, defs, equiv):
        if type(x) in internallyDefined and type(y) in internallyDefined:
            if x is y:
                raise Err("Ismorpmism of non-disjoint objects")
            elif x in defs:
                return defs[x] == y
            else:
                for fd in fields:
                    temp1 = getattr(x, fd[0])
                    temp2 = getattr(y, fd[0])
                    if type(temp1) in internallyDefined:
                        fine = temp1.__isomorphism__(temp2, defs, equiv)
                    else:
                        fine = type(x) == type(y) and equiv(x, y)
                    if not fine:
                        return fine
                    else:
                        pass
                defs[x] = y
                print(defs)
                return True
        else:
            return type(x) == type(y) and equiv(x, y)

    def isomorphism(self, other, equiv=lambda x, y: True):
        if type(other) not in internallyDefined:
            raise Err("Isomorphism not defined on external types")

        mapper = OrderedDict()
        if self.__isomorphism__(other, mapper, equiv):
            return mapper
        else:
            return None
    namespace = {}
    if namespace_injector is not None:
        namespace = namespace_injector(cname, fields, field_data, parent)
    if memoize:
        namespace["__post_init__"] = __post_init__
        namespace["__new__"] = __new__
    else:
        namespace["__post_init__"] = __post_init__
    namespace["__copy__"] = mcopy
    namespace["copy"] = mcopy
    namespace["__deepcopy__"] = dcopy
    namespace["__contains__"] = __contains__
    namespace["isdisjoint"] = isdisjoint
    namespace["isomorphism"] = isomorphism
    namespace["__isomorphism__"] = __isomorphism__

    return make_dataclass(cname, fields, bases=bases,
                          frozen=True,
                          slots=True, namespace=namespace)


def _build_classes(asdl_mod, ext_checks={},
                   ext_types={}, memoize=True,
                   namespace_injector=None, defaults={}):
    SC, ISPROD = _build_superclasses(asdl_mod)
    CHK = _build_checks(asdl_mod, SC, ext_checks)
    TYS = _build_types(SC, ext_types)
    mod = ModuleType(asdl_mod.name)
    Err = type(asdl_mod.name + " Error", (Exception,), {})
    constructorDict = {}
    fieldData = {}
    internallyDefined = set()
    for nm, t in asdl_mod.types.items():
        if isinstance(t, asdl.Product):
            fds = build_field_data(nm, t.fields, CHK, TYS, defaults)
            fieldData[nm] = fds
        elif isinstance(t, asdl.Sum):
            for c in t.types:
                fds = build_field_data(c.name, c.fields + t.attributes,
                                       CHK, TYS, defaults)
                fieldData[(nm, c.name)] = fds
        else:
            raise ADTCreationError("Unexpected asdl type: not Sum nor Product")

    def create_prod(nm, t, T):
        C = build_dc(nm, fieldData[nm], fieldData, ISPROD,
                     constructorDict, internallyDefined,
                     Err, parent=T, memoize=memoize,
                     namespace_injector=namespace_injector, defaults=defaults)
        constructorDict[nm] = C
        internallyDefined.add(C)
        return C

    def create_sum_constructor(tname, cname, T):
        C = build_dc(cname, fieldData[(tname, cname)], fieldData,
                     ISPROD, constructorDict, internallyDefined,
                     Err, parent=T, memoize=memoize,
                     namespace_injector=namespace_injector, defaults=defaults)
        constructorDict[(tname, cname)] = C
        internallyDefined.add(C)
        return C

    def create_sum(typ_name, t):
        T = SC[typ_name]
        for c in t.types:
            C = create_sum_constructor(typ_name, c.name, T)
            if hasattr(mod, c.name):
                raise ADTCreationError(f"'{c.name}' conflicts module '{mod}'")
            setattr(T, c.name, C)
            setattr(mod, c.name, C)
        return T

    for nm, t in asdl_mod.types.items():
        if isinstance(t, asdl.Product):
            setattr(mod, nm, create_prod(nm, t, SC[nm]))
        elif isinstance(t, asdl.Sum):
            setattr(mod, nm, create_sum(nm, t))
        else:
            raise ADTCreationError("Unexpected asdl type: not Sum nor Product")

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
    mod = _build_classes(asdl_ast, ext_checks=ext_checks,
                         ext_types=ext_types,
                         memoize=memoize,
                         defaults=defaults)
    # cache values in case we might want them
    mod._ext_checks = ext_checks
    mod._ext_types = ext_types
    mod._ast = asdl_ast
    mod._defstr = asdl_str
    mod._defaults = defaults
    mod._memoize = memoize
    mod.__doc__ = (f"ASDL Module Generated by ADT\n\n"
                   f"Original ASDL description:\n{asdl_str}")
    return mod
