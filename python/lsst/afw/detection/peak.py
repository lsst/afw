from __future__ import absolute_import

from lsst.afw.table.catalog import addCatalogMethods

from ._peak import PeakCatalog

__all__ = []  # import this module only for its side effects

addCatalogMethods(PeakCatalog)
