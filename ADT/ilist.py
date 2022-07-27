import typing as ty

T = ty.TypeVar('T')


class ilist(ty.Generic[T], ty.Sequence[T]):

    def __init__(self, args: ty.Iterable[T]) -> None:
        self.data = tuple(args)
        self.size = len(self.data)

    def __getitem__(self, x: int) -> T:
        return self.data[x]

    def __contains__(self, key: T) -> bool:
        return key in self.data

    def __len__(self) -> int:
        return self.size

    def __hash__(self) -> int:
        return hash(self.data)


# matching
# t = ilist([1,2,3])
# match t:
#     case [a,b]:
#         print([a,b])
#     case [a, *objects]:
#         print(objects)
#     case [a,b,c]:
#         print([a,b,c])
#     case _:
#         print("oops")
# #implicit conversion
