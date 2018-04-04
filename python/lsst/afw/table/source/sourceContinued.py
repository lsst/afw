#
# LSST Data Management System
# Copyright 2017 LSST/AURA.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
from __future__ import absolute_import, division, print_function

__all__ = []

from lsst.utils import continueClass
from ..base import Catalog
from .source import SourceCatalog, SourceTable

Catalog.register("Source", SourceCatalog)


@continueClass  # noqa F811
class SourceCatalog:

    def getChildren(self, parent, *args):
        """Return the subset of self for which the parent field equals the given value.

        In order for this method to return the correct result, it must be sorted by parent
        (i.e. self.isSorted(SourceTable.getParentKey()) must be True).  This is naturally the
        case with SourceCatalogs produced by the detection and deblending tasks, but it may
        not be true when concatenating multiple such catalogs.

        Additional Catalogs or sequences whose elements correspond in order to the records
        of self (i.e. zip(self, *args) is valid) will be subset using the same slice object
        used on self, and these subsets will be returned along with the subset of self.
        """
        if not self.isSorted(SourceTable.getParentKey()):
            raise AssertionError(
                "The table is not sorted by parent, so cannot getChildren")
        s = self.equal_range(parent, SourceTable.getParentKey())
        if args:
            return (self[s],) + tuple(arg[s] for arg in args)
        else:
            return self[s]
