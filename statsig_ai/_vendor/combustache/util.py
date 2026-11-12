from os import PathLike
from pathlib import Path
from typing import Any, Callable, TypedDict

StrPath = PathLike[str] | str

LAMBDA = '<lambda>'


class Opts(TypedDict):
    """
    `render` and `Template.render` options.
    """

    stringify: Callable[[Any], str]
    escape: Callable[[str], str]
    missing_data: Callable[[], Any]


def is_whitespace(string: str) -> bool:
    """
    Checks if a string is whitespace.
    """
    # ''.isspace() returns False
    # for our purposes if there is nothing after or behind the tag
    # it still should count as whitespace and the tag should be standalone
    return not string or string.isspace()


def to_str(val: Any) -> str:
    """
    Turns a value into a string with None -> ''.

    Args:
        val: Value to stringify.

    Returns:
        Stringified value.
    """
    if val is None:
        return ''
    return str(val)


def find_position(template: str, index: int) -> tuple[int, int]:
    """
    Finds row and column of an index in a template.

    Args:
        template: Template.
        index: Index.

    Returns a tuple (row, col) where:
        row: Row number.
        col: Column number.
    """
    row = template.count('\n', 0, index)
    last_break = template.rfind('\n', 0, index)
    col = index - (last_break + 1)
    return row + 1, col + 1


def find_template_files(path: Path, extension: str) -> list[Path]:
    return list(path.rglob(f'**/*{extension}'))


def paths_to_templates(
    partial_paths: list[Path],
    extension: str,
    include_relative_path: bool,
    relative_to: Path,
) -> dict[str, str]:
    if include_relative_path:
        return {
            '.'.join(path.relative_to(relative_to).parts).removesuffix(
                extension
            ): path.read_text()
            for path in partial_paths
        }
    return {
        path.name.removesuffix(extension): path.read_text()
        for path in partial_paths
    }


def load_templates(
    path: StrPath, extension: str, *, include_relative_path: bool = False
) -> dict[str, str]:
    """
    Loads templates from a directory.

    Example::

        my_project/
        ├─ templates/
        │  ├─ ui/
        │  │  └─ comment.mustache
        │  ├─ index.mustache
        │  └─ other.file
        └─ ...
        >>> load_templates('./templates/', '.mustache')
        {
            'comment': '<div class='comment'> {{content}} </div>',
            'index': '<h1> Welcome, {{username}}! </h1>',
        }
        >>> load_templates('./templates/', '.mustache',
        ... include_relative_path=True)
        {
            'ui.comment': '<div class='comment'> {{content}} </div>',
            'index': '<h1> Welcome, {{username}}! </h1>',
        }

    Args:
        path: Root directory path.
        extension: Template file extension.
        include_relative_path: Include template's relative path in its name.

    Returns:
        Dictionary of templates.
    """
    path = Path(path)
    partial_paths = find_template_files(path, extension)
    return paths_to_templates(
        partial_paths, extension, include_relative_path, path
    )
