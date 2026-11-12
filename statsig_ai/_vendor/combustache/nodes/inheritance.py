import textwrap

from .. import main
from ..ctx import MISSING, Ctx
from ..util import Opts, is_whitespace
from .node import Node
from .section import Section


def is_paired(node1: Node, node2: Node) -> bool:
    return (
        is_whitespace(node1.before)
        and is_whitespace(node2.after)
        and node1.tag_end == node2.tag_start
    )


class Block(Section):
    left = '$'

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
        if is_paired(self, self.closing_tag):
            self.is_standalone = True
            self.actual_start = self.line_start
            self.closing_tag.actual_end = self.closing_tag.line_end
            # standalone blocks are unique
            # https://github.com/mustache/spec/pull/131#discussion_r668359813
            #    if the end section tag is not standalone by itself, then
            #    the trailing newline will not be removed from the output
            #    if the block ends up being empty
            self.closing_tag.actual_end -= 1

        self.set_indentation_and_default_value()

    def set_indentation_and_default_value(self):
        text = self.inside_text

        if self.is_standalone:
            dedented_text = textwrap.dedent(text)
            dedented_line = dedented_text.split('\n')[0]
            full_line = text.split('\n')[0]

            self.indent = full_line.removesuffix(dedented_line)
            self.default_value = dedented_text
        else:
            self.indent = None
            self.default_value = text

    def handle(self, ctx: Ctx, partials: dict[str, str], opts: Opts) -> str:
        data = ctx.inheritance_args.get(self.contents, MISSING)

        if data is MISSING:
            partial_template = self.default_value
        else:
            partial_template = data

        if self.is_standalone:
            partial_template = textwrap.indent(
                partial_template, self.indent or self.before
            )
        return main._render(partial_template, ctx, partials, opts)


class Parent(Section):
    left = '<'

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
        if is_paired(self, self.closing_tag):
            self.is_standalone = True
            self.actual_start = self.line_start
            self.closing_tag.actual_end = self.closing_tag.line_end
        self.blocks = [v for v in self.inside._list if isinstance(v, Block)]

        if self.blocks:
            # {{< parent}}{{$ block}} are pair standalone
            first_block = self.blocks[0]
            if is_paired(self, first_block):
                self.is_standalone = True
                first_block.is_standalone = True

                self.practical_start = self.line_start
                first_block.actual_end = first_block.line_end
                first_block.set_indentation_and_default_value()

            # {{/ parent}}{{/ block}} are pair standalone
            last_block = self.blocks[-1]
            if is_paired(last_block.closing_tag, self.closing_tag):
                last_block.closing_tag.actual_start = (
                    last_block.closing_tag.line_start
                )
                self.closing_tag.actual_end = self.closing_tag.line_end
                last_block.set_indentation_and_default_value()

    def handle(self, ctx: Ctx, partials: dict[str, str], opts: Opts) -> str:
        missing_data = opts['missing_data']

        for block in self.blocks:
            ctx.inheritance_args.setdefault(
                block.contents, block.default_value
            )

        partial_template = partials.get(self.contents)

        if partial_template is None:
            ctx.inheritance_args.clear()
            return missing_data()

        if self.is_standalone or self.is_pair_standalone:
            partial_template = textwrap.indent(partial_template, self.before)

        res = main._render(partial_template, ctx, partials, opts)
        ctx.inheritance_args.clear()
        return res
