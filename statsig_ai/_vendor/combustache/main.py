import html
from typing import Any, Callable, Type

from .ctx import Ctx
from .nodes import (
    Ampersand,
    Block,
    Closing,
    Comment,
    Delimiter,
    Interpolation,
    Inverted,
    Node,
    Parent,
    Partial,
    Section,
    Triple,
)
from .util import Opts, to_str

_node_types: dict[str, type[Node]] = {
    node.left: node
    for node in {
        Comment,
        Section,
        Ampersand,
        Inverted,
        Closing,
        Triple,
        Partial,
        Delimiter,
        Block,
        Parent,
    }
}


def find_node(
    template: str,
    search_start: int,
    template_end: int,
    left_delimiter: str,
    right_delimiter: str,
) -> tuple[Type[Node], str, int, int] | None:
    """
    Finds the first node from the search_start in a template.

    Returns a tuple (NodeType, contents, left_outside_idx, right_outside_idx)
    or None where:
        NodeType: Type of the found Node.
        contents: Inside contents of the found node.
        left_outside_idx: Index where the node starts.
        right_outside_idx: Index where the node ends.
    """
    # Hello {{> hello.world }}!
    #       ^ ^^            ^ ^
    #       1 23            4 5
    # 1. left_outside_idx
    # 2. left_char_idx
    # 3. left_inside_idx
    # 4. right_inside_idx
    # 5. right_outside_idx
    left_outside_idx = template.find(
        left_delimiter, search_start, template_end
    )
    if left_outside_idx < search_start:
        return None

    left_char_idx = left_outside_idx + len(left_delimiter)
    if left_char_idx >= template_end:
        return None

    NodeType = _node_types.get(template[left_char_idx], Interpolation)
    node_left_char = NodeType.left
    node_right_char = NodeType.right

    left_inside_idx = left_char_idx + len(node_left_char)

    new_right_delimiter = node_right_char + right_delimiter
    right_inside_idx = template.find(
        new_right_delimiter, left_char_idx, template_end
    )
    if right_inside_idx < left_inside_idx:
        return None

    right_outside_idx = right_inside_idx + len(new_right_delimiter)
    contents = template[left_inside_idx:right_inside_idx].strip()
    return NodeType, contents, left_outside_idx, right_outside_idx


class Template:
    """
    Mustache template.
    """

    def __init__(
        self,
        template: str,
        left_delimiter: str = '{{',
        right_delimiter: str = '}}',
        template_start: int = 0,
        template_end: int | None = None,
    ) -> None:
        """
        Initializes a mustache template.

        Args:
            template: Mustache template.
            left_delimiter: Left tag delimiter.
            right_delimiter: Right tag delimiter.
            template_start: Template start index.
            template_end: Template end index.

        Raises:
            DelimiterError: Bad delimiter tag.
            MissingClosingTagError: Missing closing tag.
            StrayClosingTagError: Stray closing tag.
        """
        if template_end is None:
            template_end = len(template)
        parsed_template = []
        search_start = template_start
        while True:
            node_info = find_node(
                template,
                search_start,
                template_end,
                left_delimiter,
                right_delimiter,
            )

            if node_info is None:
                parsed_template.append(template[search_start:template_end])
                break

            NodeType, contents, start, end = node_info
            node = NodeType(
                contents,
                start,
                end,
                template,
                template_start,
                template_end,
                left_delimiter,
                right_delimiter,
            )

            if NodeType is Delimiter:
                left_delimiter = node.left_delimiter
                right_delimiter = node.right_delimiter

            parsed_template.append(template[search_start : node.actual_start])
            if not node.ignorable:
                parsed_template.append(node)
            search_start = node.parse_end

        self._list: list[Node | str] = parsed_template

    def _render(self, ctx: Ctx, partials: dict[str, str], opts: Opts) -> str:
        return ''.join(
            node.handle(ctx, partials, opts)
            if isinstance(node, Node)
            else node
            for node in self._list
        )

    def render(
        self,
        data: dict[str, Any],
        partials: dict[str, str] | None = None,
        *,
        stringify: Callable[[Any], str] | None = None,
        escape: Callable[[str], str] | None = None,
        missing_data: Callable[[], Any] | None = None,
    ) -> str:
        """
        Renders a mustache template.

        Args:
            data: Values to insert into the template.
            partials: Partials to insert into the template.

        Keyword args:
            stringify: String conversion function.
            escape: Escaping function.
            missing_data: Function called on missing data.

        Returns:
            Rendered template.
        """
        opts: Opts = {
            'stringify': stringify or to_str,
            'escape': escape or html.escape,
            'missing_data': missing_data or (lambda: ''),
        }
        if partials is None:
            partials = {}
        ctx = Ctx([data])
        return self._render(ctx, partials, opts)


def _render(
    template: str,
    ctx: Ctx,
    partials: dict[str, str],
    opts: Opts,
    left_delimiter: str = '{{',
    right_delimiter: str = '}}',
) -> str:
    root = Template(template, left_delimiter, right_delimiter)
    return root._render(ctx, partials, opts)


def render(
    template: str,
    data: dict[str, Any],
    partials: dict[str, str] | None = None,
    left_delimiter: str = '{{',
    right_delimiter: str = '}}',
    *,
    stringify: Callable[[Any], str] | None = None,
    escape: Callable[[str], str] | None = None,
    missing_data: Callable[[], Any] | None = None,
) -> str:
    """
    Renders a mustache template.

    Args:
        template: Mustache template.
        data: Values to insert into the template.
        partials: Partials to insert into the template.
        left_delimiter: Left tag delimiter.
        right_delimiter: Right tag delimiter.

    Keyword args:
        stringify: String conversion function.
        escape: Escaping function.
        missing_data: Function called on missing data.

    Returns:
        Rendered template.

    Raises:
        DelimiterError: Bad delimiter tag.
        MissingClosingTagError: Missing closing tag.
        StrayClosingTagError: Stray closing tag.
    """
    opts: Opts = {
        'stringify': stringify or to_str,
        'escape': escape or html.escape,
        'missing_data': missing_data or (lambda: ''),
    }
    if partials is None:
        partials = {}
    ctx = Ctx([data])
    return _render(
        template, ctx, partials, opts, left_delimiter, right_delimiter
    )
