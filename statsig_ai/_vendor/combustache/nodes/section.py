from typing import Iterable, Mapping

from .. import main
from ..ctx import MISSING, Ctx
from ..exceptions import MissingClosingTagError, StrayClosingTagError
from ..util import LAMBDA, Opts, find_position
from .node import Node


class Section(Node):
    left = '#'

    def __init__(
        self,
        contents: str,
        tag_start: int,
        tag_end: int,
        template: str,
        template_start: int,
        template_end: int,
        left_delimiter: str,
        right_delimiter: str,
    ) -> None:
        super().__init__(
            contents,
            tag_start,
            tag_end,
            template,
            template_start,
            template_end,
            left_delimiter,
            right_delimiter,
        )
        depth = 0
        search_start = self.actual_end
        while True:
            node_info = main.find_node(
                self.template,
                search_start,
                self.template_end,
                self.left_delimiter,
                self.right_delimiter,
            )
            if node_info is None:
                row, col = find_position(self.template, self.tag_start)
                raise MissingClosingTagError(
                    f'No closing tag found: {self.tag_string} at {row}:{col}'
                )

            NodeType, contents, start, end = node_info
            if contents == self.contents:
                if NodeType is self.__class__:
                    depth += 1
                elif NodeType is Closing:
                    if depth == 0:
                        break
                    depth -= 1

            search_start = end

        # not creating a ClosingTag because it raises the stray tag error
        closing_tag = Node(
            contents,
            start,
            end,
            self.template,
            self.template_start,
            self.template_end,
            self.left_delimiter,
            self.right_delimiter,
        )
        self.closing_tag = closing_tag

        self.inside = main.Template(
            self.template,
            self.left_delimiter,
            self.right_delimiter,
            self.actual_end,
            self.closing_tag.actual_start,
        )

    @property
    def parse_end(self) -> int:
        return self.closing_tag.actual_end

    @property
    def inside_text(self) -> str:
        return self.template[self.actual_end : self.closing_tag.actual_start]

    def should_be_rendered(self, item):
        return (
            item
            and item is not MISSING
            and (not isinstance(item, Iterable) or len(list(item)))
        )

    def handle(self, ctx: Ctx, partials: dict[str, str], opts: Opts) -> str:
        missing_data = opts['missing_data']

        data = ctx.get(self.contents)

        if not self.should_be_rendered(data):
            if data is MISSING:
                return missing_data()
            return ''

        if callable(data):
            unprocessed = self.inside_text
            # if the callable is a lambda we should call it with the string
            # between the tag and its closing tag and render the result
            # in the current context
            if data.__name__ == LAMBDA:
                template = str(data(unprocessed))
                return main._render(
                    template,
                    ctx,
                    partials,
                    opts,
                    self.left_delimiter,
                    self.right_delimiter,
                )
            else:
                # otherwise we should get the result with the string passed in
                try:
                    data = data(unprocessed)
                # or just called
                except TypeError:
                    data = data()

        # if data is not an iterable we put it into a list
        # so we can process it easily
        if not (
            isinstance(data, Iterable)
            and not isinstance(data, str | bytes | Mapping)
        ):
            data = [data]

        handled = []
        for item in data:
            ctx.append(item)
            handled.append(self.inside._render(ctx, partials, opts))
            ctx.pop()
        return ''.join(handled)


class Inverted(Section):
    left = '^'

    def should_be_rendered(self, item):
        return not super().should_be_rendered(item)

    def handle(self, ctx: Ctx, partials: dict[str, str], opts: Opts) -> str:
        missing_data = opts['missing_data']

        data = ctx.get(self.contents)

        if not self.should_be_rendered(data):
            if data is MISSING:
                return missing_data()
            return ''

        return self.inside._render(ctx, partials, opts)


class Closing(Node):
    left = '/'
    ignorable = True

    # Section and Inverted find their closing tags themselves
    # so if the parser tries to create a Closing tag that means that the tag
    # was not opened to be closed
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        row, col = find_position(self.template, self.tag_start)
        raise StrayClosingTagError(
            f'Stray closing tag found: {self.tag_string} at {row}:{col}'
        )
