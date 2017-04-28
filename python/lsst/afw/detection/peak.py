from __future__ import absolute_import, division, print_function

from lsst.afw.table import Catalog

from ._peak import PeakCatalog

__all__ = []  # import this module only for its side effects

Catalog.register("Peak", PeakCatalog)
