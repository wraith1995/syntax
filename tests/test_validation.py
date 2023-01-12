"""
Tests of validator edge cases
"""

# pylint: disable=no-member
import re
from enum import Enum
from typing import Type, List, Literal

import pytest

import adt

def test_object_is_not_none():
    """
    Test that "object"-typed fields may not be none
    """
    test_adt = adt.ADT("module test_object_is_not_none { foo = ( object x ) }")

    assert isinstance(test_adt.foo(3), test_adt.foo)
    assert isinstance(test_adt.foo("bar"), test_adt.foo)

    with pytest.raises(test_adt.__err__):
        test_adt.foo(None)
def test_optional_may_be_none():
    """
    Test that _optional_ "object"-typed fields _may_ be none
    """
    test_adt = adt.ADT("module test_optional_may_be_none { foo = ( object? x ) }")
    assert isinstance(test_adt.foo(None), test_adt.foo)
    assert test_adt.foo(None).x is None

# FIXME: This seems like an insane idea given how types currently work in python
# def test_subclass_validator():
#     """
#     Test that Type[X] accepts class values which are subtypes of X.
#     """

#     # pylint: disable=too-few-public-methods,missing-class-docstring
#     class Parent:
#         pass

#     class Child(Parent):
#         pass

#     test_adt = adt.ADT(
#         "module test_subclass_validator { foo = ( parent x ) }",
#         ext_types={"parent": Type[Parent]},
#     )

#     assert isinstance(test_adt.foo(Parent), test_adt.foo)
#     assert isinstance(test_adt.foo(Child), test_adt.foo)

#     with pytest.raises(test_adt.__err__) as exc_info:
#         test_adt.foo(Parent())

#     # assert exc_info.value.expected == Type[Parent]
#     # assert exc_info.value.actual == Parent    
