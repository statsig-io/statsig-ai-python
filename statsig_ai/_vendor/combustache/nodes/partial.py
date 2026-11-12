from .. import main
from ..ctx import Ctx
from ..util import Opts
from .node import Node


class Partial(Node):
    left = '>'

    def handle(self, ctx: Ctx, partials: dict[str, str], opts: Opts) -> str:
        missing_data = opts['missing_data']

        if self.contents[0] == '*':
            partial_name = ctx.get(self.contents[1:].strip())
            partial_template = partials.get(partial_name)
        else:
            partial_template = partials.get(self.contents)

        if partial_template is None:
            return missing_data()

        if self.is_standalone:
            partial_template = '\n'.join(
                bool(line) * self.before + line
                for line in partial_template.split('\n')
            )

        return main._render(partial_template, ctx, partials, opts)
