from .node import Node


class Comment(Node):
    left = '!'
    ignorable = True
