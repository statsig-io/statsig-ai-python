class CombustacheError(Exception):
    """
    Combustache base exception. Not raised.
    """

    pass


class DelimiterError(CombustacheError):
    """
    A delimiter tag contents are wrong.
    """

    pass


class MissingClosingTagError(CombustacheError):
    """
    A closing tag for an opened tag is not found.
    """

    pass


class StrayClosingTagError(CombustacheError):
    """
    A closing tag for an unopened tag is found.
    """

    pass
