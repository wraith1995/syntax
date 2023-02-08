"""Printing system for an ADT Module."""
from typing import List, NamedTuple, Optional, Tuple
import pprint
from .auxtypes import stamp
from string import Formatter


class ExprPrint(NamedTuple):
    """Specification for how to print an expression."""

    oper: str
    meta: Optional[str]
    exprs: str


class HeaderBlockStatement(NamedTuple):
    """Specification for how to print a statement with a potential block."""

    statement: str
    meta: Optional[str]
    nest: Optional[Tuple[str, str]]
    rest: List[str]


class StmtPrint(NamedTuple):
    """Specification for how to print a statement."""

    top: HeaderBlockStatement
    rest: List[HeaderBlockStatement]


# PLAN:
# Add an other option: Just a format string
# BUILD MATCH INFUSTRUCTURE
# Build _pprint_operation attachement.
# Auto GEN given expr/statement classification.
# pintegrate into format by gen _print_operation:
# https://stackoverflow.com/questions/3258072/best-way-to-implement-custom-pretty-printers
# Add appropriate dispatch into these based on match propsoal: match one, call matched _pprint, ...
# do it.


# class printContext(object):
#     """Hack to carry context in pretty printer."""

#     def __init__(self, context=dict(), depthDict=dict(), stampFlag: bool = True, other: Any = None):
#         """Create a default context."""
#         self.context = context
#         self.depthDict: Dict[Any, int] = dict()
#         self.stampFlag = stampFlag
#         self.other = other

#     # set item
#     # contains
#     # del item


# def build_printing_class(
#     module: ModuleType,
#     isSeq: Callable[[str, T], bool],
#     isOpt: Callable[[str, T], bool],
#     moduleType: Type[T],
#     isExpr: Callable[[T], bool],
#     exprToOpt: Callable[[T], str],
#     exprToData: Callable[[T], FMT],
#     exprToArgs: Callable[[T], FMT],
#     lhsAssocitate: bool,
#     exprToPrec: Callable[[T], int],
#     isStmt: Callable[[T], bool],
#     stmtParts: Callable[[T], Sequence[Tuple[str, FMT, Sequence[T]]]],  # name, meta, sub stmts
#     isGraph: Optional[Callable[[T], bool]],
# ) -> pprint.PrettyPrinter:
#     """Create a pretty printer for ADT Module."""
#     class ModulePrinter(pprint.PrettyPrinter):
#         def format(self, object, stream, indent, allowance, context, level):


#     raise NotImplementedError("oops")
