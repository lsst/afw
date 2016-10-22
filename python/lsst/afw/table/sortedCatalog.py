from __future__ import absolute_import, division, print_function

from .catalog import addCatalogMethods

__all__ = ["addSortedCatalogMethods"]


def addSortedCatalogMethods(cls):
    """Add pure python methods to SortedCatalogT classes or subclasses

    This should be called for most catalog classes except BaseCatalog,
    including SimpleCatalog and SourceCatalog.

    Note: this calls `addCatalogMethods`.
    """
    addCatalogMethods(cls)

    def _isSorted(self, key=None):
        if key is None:
            key = self.getIdKey()
        return type(self).__base__.isSorted(self, key)

    def _sort(self, key=None):
        if key is None:
            key = self.getIdKey()
        type(self).__base__.sort(self, key)

    cls.isSorted = _isSorted
    cls.sort = _sort
