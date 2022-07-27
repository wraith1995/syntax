
stampsdict = dict()


class stamp:
    __slots__ = ('__weakref__', 'stamp', "args")

    def __init__(self, args=None):
        h = hash(args)
        if h in stampsdict:
            stampsdict[h] += 1
        else:
            stampsdict[h] = 1
        self.stamp = stampsdict[h]
        self.args = args

    def __hash__(self):
        return self.stamp

    def __repr__(self):
        return "@{0}".format(self.stamp)

    def __copy__(self):
        return stamp(args=self.args)

    def __deepcopy__(self):
        return stamp(args=self.args)


def defaultStamp():
    return stamp()
