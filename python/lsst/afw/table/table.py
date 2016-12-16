from __future__ import absolute_import, division, print_function

__all__ = ["addTableMethods"]


def addTableMethods(cls):
    """Add pure python methods to a table class

    This should be called for BaseTable and all subclasses of BaseTable
    """
    def cast(self, type_):
        """Cast this table to the specified type
        """
        return type_._castFrom(self)

    cls.cast = cast
