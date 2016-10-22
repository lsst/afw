from __future__ import absolute_import, division, print_function

from .sortedCatalog import addSortedCatalogMethods
from ._simpleCatalog import SimpleCatalog

__all__ = []  # import this module only for its side effects

addSortedCatalogMethods(SimpleCatalog)
