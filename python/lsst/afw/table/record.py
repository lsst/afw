from __future__ import absolute_import, division, print_function

__all__ = ["addRecordMethods"]


def addRecordMethods(cls):
    """Add pure python methods to a record class

    This should be called BaseReocrd and all subclasses of BaseRecord
    """
    def cast(self, type_):
        """Cast this record to the specified type
        """
        return type_._castFrom(self)

    cls.cast = cast
