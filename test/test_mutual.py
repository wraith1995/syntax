import pytest
from adt import ADT


def test_sum_mutual():
    try:
        return ADT("""
        module Test {
        test1 = A (test2 t) | B (test1 tt) | C (int ttt)
        test2 = D (test1 tttt) | E (test2 ttttt) | F (int tttttt)
        }
        """)
    except Exception as exc:
        assert False, str(exc)

def test_prod_mutual():
    try:
        return ADT("""
        module Test {
        test1 = (int a, test2 b)
        test2 = (int c, test1 d)
        }
        """)
    except Exception as exc:
        assert False, str(exc)

def test_mixed_mutual():
    try:
        return ADT("""
        module Test {
        test1 = (int a, test2 b)
        test2 = A(int c, test1 d)
        }
        """)
    except Exception as exc:
        assert False, str(exc)
