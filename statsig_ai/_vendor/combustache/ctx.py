from typing import Any

MISSING = object()


class Ctx:
    """
    Context stack.
    """

    def __init__(self, stack: list[Any]):
        self.stack = stack
        self.inheritance_args = {}

    def pop(self, index: int = -1) -> Any:
        return self.stack.pop(index)

    def append(self, value: Any) -> None:
        return self.stack.append(value)

    def get(self, key: str) -> Any:
        """
        Gets a value from a context with a key.

        If nothing was found, combustache.ctx.MISSING is returned.

        Args:
            key: Key.

        Returns:
            Value or MISSING.
        """
        if key == '.':
            return self.stack[-1]

        chain = key.split('.')

        found = self.find_first(chain[0])
        if found is MISSING:
            return MISSING

        for key in chain[1:]:
            found = self.deep_get(found, key)
        return found

    def find_first(self, key: str):
        rctx = reversed(self.stack)
        for item in rctx:
            found = self.deep_get(item, key)
            if found is not MISSING:
                return found
        return MISSING

    @staticmethod
    def deep_get(item: Any, key: str) -> Any:
        # if the item is a callable we should be retrieving from its result
        try:
            item = item()
        except TypeError:
            pass

        # try getting a value from a Mapping
        try:
            return item[key]
        except KeyError:
            return MISSING
        except TypeError:
            pass

        # try indexing into the item like a Sequence
        try:
            idx = int(key)
            return item[idx]
        except IndexError:
            return MISSING
        except ValueError:
            pass

        # simple getattr for all other cases
        return getattr(item, key, MISSING)
