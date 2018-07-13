__all__ = []  # import this module only for its side effects

from lsst.utils import continueClass
from .ellipse import Ellipse


@continueClass  # noqa F811
class Ellipse:
    def __repr__(self):
        return "Ellipse(%r, %r)" % (self.getCore(), self.getCenter())

    def __reduce__(self):
        return (Ellipse, (self.getCore(), self.getCenter()))

    def __str__(self):
        return "(%s, %s)" % (self.getCore(), self.getCenter())
