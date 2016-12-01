from __future__ import absolute_import, division, print_function

from .catalog import addCatalogMethods
from ._exposure import ExposureCatalog

__all__ = []  # import this module only for its side effects

addCatalogMethods(ExposureCatalog)
