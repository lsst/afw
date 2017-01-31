from __future__ import absolute_import, division, print_function

from .catalog import addCatalogMethods, searchTemplate

__all__ = ["addSortedCatalogMethods"]


def addSortedCatalogMethods(cls):
    """Add pure python methods to SortedCatalogT classes or subclasses

    This should be called for most catalog classes except BaseCatalog,
    including SimpleCatalog and SourceCatalog.

    Note: this calls `addCatalogMethods`.
    """
    addCatalogMethods(cls)

    def isSorted(self, key=None):
        if key is None:
            key = self.table.getIdKey()
        return type(self).__base__.isSorted(self, key)
    cls.isSorted = isSorted

    # emulate sort() using sort(key) to simplify python argument lookup
    def sort(self, key=None):
        if key is None:
            key = self.table.getIdKey()
        self._sort(key)
    cls.sort = sort

    def find(self, value, key=None):
        """Return the record for which record.get(key) == value

        If no such record is found, return None; if multiple such records are found,
        return one of them.

        The catalog must be sorted by this key before this method is called.
        """
        if key is None:
            key = self.table.getIdKey()
        return searchTemplate(self, "_find_", value, key)
    cls.find = find
