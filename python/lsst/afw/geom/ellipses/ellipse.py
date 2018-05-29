from ._ellipse import Ellipse

__all__ = []  # import this module only for its side effects


def __repr__(self):
    return "Ellipse(%r, %r)" % (self.getCore(), self.getCenter())


Ellipse.__repr__ = __repr__


def __reduce__(self):
    return (Ellipse, (self.getCore(), self.getCenter()))


Ellipse.__reduce__ = __reduce__


def __str__(self):
    return "(%s, %s)" % (self.getCore(), self.getCenter())


Ellipse.__str__ = __str__
