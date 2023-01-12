"""
Tests of the sum-type "attributes" feature of ASDL.
"""

# pylint: disable=no-member

import ADT as adt


def test_basic_attributes() -> None:
    """
    Test a simple attributes scenario... no memoization or custom types.
    """
    test_adt = adt.ADT(
        """
        module test_adt {
            sum = A()
                | B( int x )
                attributes( int y, int z )
        }
        """
    )
    assert isinstance(test_adt.A(3, 4), test_adt.A)
