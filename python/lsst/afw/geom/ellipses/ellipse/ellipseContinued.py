__all__ = []  # import this module only for its side effects

from lsst.utils import continueClass
from .ellipse import Ellipse


@continueClass  # noqa: F811
class Ellipse:
    def __repr__(self):
        return f"Ellipse({self.getCore()!r}, {self.getCenter()!r})"

    def __reduce__(self):
        return (Ellipse, (self.getCore(), self.getCenter()))

    def __str__(self):
        return f"({self.getCore()}, {self.getCenter()})"
