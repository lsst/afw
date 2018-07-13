__all__ = []  # import this module only for its side effects

from lsst.afw.table import Catalog
from .peak import PeakCatalog

Catalog.register("Peak", PeakCatalog)
