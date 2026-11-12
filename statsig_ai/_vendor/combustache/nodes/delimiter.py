from ..exceptions import DelimiterError
from ..util import find_position
from .node import Node


class Delimiter(Node):
    left = '='
    right = '='
    ignorable = True

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
        split = self.contents.split()
        if len(split) != 2:
            row, col = find_position(self.template, self.tag_start)
            raise DelimiterError(
                'Impossible delimiter tag found: '
                f'{self.tag_string} at {row}:{col}'
            )
        self.left_delimiter = split[0]
        self.right_delimiter = split[1]
