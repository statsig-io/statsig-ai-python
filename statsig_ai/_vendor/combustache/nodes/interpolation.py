from typing import Any, Callable

from .. import main
from ..ctx import MISSING, Ctx
from ..util import LAMBDA, Opts
from .node import Node


class Interpolation(Node):
    standalonable = False

    def get_string(
        self,
        data: Any,
        stringify: Callable[[Any], str],
        escape: Callable[[str], str],
    ) -> str:
        return escape(stringify(data))

    def handle(self, ctx: Ctx, partials: dict[str, str], opts: Opts) -> str:
        stringify = opts['stringify']
        escape = opts['escape']
        missing_data = opts['missing_data']

        data = ctx.get(self.contents)
        if data is MISSING:
            return missing_data()

        if callable(data):
            # if the callable is a lambda function we should render its result
            # in the current context
            if data.__name__ == LAMBDA:
                template = str(data())
                data = main._render(template, ctx, partials, opts)
            # otherwise we should just get the result
            else:
                data = data()

        string = self.get_string(data, stringify, escape)
        return string


class Ampersand(Interpolation):
    left = '&'

    def get_string(
        self,
        data: Any,
        stringify: Callable[[Any], str],
        escape: Callable[[str], str],
    ) -> str:
        return stringify(data)


class Triple(Ampersand):
    left = '{'
    right = '}'
