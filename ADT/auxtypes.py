"""Auxiliary types useful for making IRs."""
from collections.abc import MutableMapping

stampsdict: MutableMapping[int, int] = dict()


class stamp:
    """A class representing a unique name."""

    __slots__ = ("__weakref__", "stamp", "args")

    def __init__(self, args=None):
        """Create a unique name."""
        h = hash(args)
        if h in stampsdict:
            stampsdict[h] += 1
        else:
            stampsdict[h] = 1
        self.stamp = stampsdict[h]
        self.args = args

    def __hash__(self):
        """Return the stamp."""
        return self.stamp

    def __repr__(self):
        """Represent the stamp."""
        return "@{0}".format(self.stamp)

    def __copy__(self):
        """Copy the stamp."""
        return stamp(args=self.args)

    def __deepcopy__(self):
        """Deep copy of the stamp."""
        return stamp(args=self.args)

    def __lt__(self, other):
        """Compare two stamps."""
        assert isinstance(other, stamp)
        return self.stamp < other.stamp


def defaultStamp():
    """Create the empty stamp."""
    return stamp()
