from __future__ import annotations
from adt import ADT
from snake_egg import EGraph, Rewrite, Var, vars
import pytest

math = ADT(
    """ module MATH{
const = (int int)
expr = Add(expr x, expr y) | Mul(expr x, expr y) | Const (const x) | Paren (expr x)
    }
""",
    ext_types={"int": int},
    egraphableTypes=True,
    slots=False,
)


a = Var("a")
b = Var("b")
q = math.Add(b, math.Const(0))
rules = [
    Rewrite(math.Add(a, b), math.Add(b, a), name="commute-add"),
    Rewrite(math.Mul(a, b), math.Mul(b, a), name="commute-mul"),
    Rewrite(q, math.Paren(b), name="add-0"),
    # Rewrite(math.Paren(a), a, name="umm"),
    Rewrite(math.Mul(a, math.Const(0)), math.Const(0), name="mul-0"),
    Rewrite(math.Mul(a, math.Const(1)), math.Paren(a), name="mul-1"),
]


def simplify(expr, iters=7):
    egraph = EGraph()
    egraph.add(expr)
    egraph.run(rules, iters)
    best = egraph.extract(expr)
    print(best)
    return best


def checkExists(expr, expr1, iters=7):
    egraph = EGraph()
    egraph.add(expr)
    egraph.run(rules, iters)
    best = egraph.extract(expr1)
    print(best)
    return best


def test_simple_1():
    assert checkExists(
        math.Mul(math.Const(0), math.Const(42)), math.Paren(math.Const(0))
    ) == math.Paren(math.Const(0))


def test_simple_2():
    foo = Var("foo")
    assert checkExists(
        math.Add(math.Const(0), math.Mul(math.Const(1), foo)), math.Paren(foo)
    ) == math.Paren(foo)
