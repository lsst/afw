from __future__ import absolute_import

from lsst.afw.table.catalog import addCatalogMethods

from ._peak import PeakRecordCatalog

__all__ = [] # only imported for side effects

addCatalogMethods(PeakRecordCatalog)
