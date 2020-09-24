# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = []

from lsst.utils import continueClass
from lsst.utils.deprecated import deprecate_pybind11
from ._base import Catalog
from ._table import SourceCatalog, SourceTable

Catalog.register("Source", SourceCatalog)


@continueClass  # noqa: F811 (FIXME: remove for py 3.8+)
class SourceCatalog:  # noqa: F811

    def getChildren(self, parent, *args):
        """Return the subset of self for which the parent field equals the
        given value.

        In order for this method to return the correct result, it must be
        sorted by parent (i.e. self.isSorted(SourceTable.getParentKey()) must
        be True).  This is naturally the case with SourceCatalogs produced by
        the detection and deblending tasks, but it may not be true when
        concatenating multiple such catalogs.

        Additional Catalogs or sequences whose elements correspond in order to
        the records of self (i.e. ``zip(self, *args)`` is valid) will be
        subset using the same slice object used on self, and these subsets
        will be returned along with the subset of self.

        Parameters
        ----------
        parent : `int`
            ID of the parent to get children for.
        args : `~lsst.afw.table.Catalog`
            Additional catalogs to subset for the childrens to return.

        Returns
        -------
        children : iterable of `~lsst.afw.table.SourceRecord`
            Children sources.
        """
        if not self.isSorted(SourceTable.getParentKey()):
            raise AssertionError(
                "The table is not sorted by parent, so cannot getChildren")
        s = self.equal_range(parent, SourceTable.getParentKey())
        if args:
            return (self[s],) + tuple(arg[s] for arg in args)
        else:
            return self[s]


SourceTable.getCentroidDefinition = deprecate_pybind11(
    SourceTable.getCentroidDefinition,
    reason='Use `getSchema().getAliasMap().get("slot_Centroid")` instead. To be removed after 20.0.0.')
SourceTable.hasCentroidSlot = deprecate_pybind11(
    SourceTable.hasCentroidSlot,
    reason='Use `getCentroidSlot().isValid()` instead. To be removed after 20.0.0.')
SourceTable.getCentroidKey = deprecate_pybind11(
    SourceTable.getCentroidKey,
    reason='Use `getCentroidSlot().getMeasKey()` instead. To be removed after 20.0.0.')
SourceTable.getCentroidErrKey = deprecate_pybind11(
    SourceTable.getCentroidErrKey,
    reason='Use `getCentroidSlot().getErrKey()` instead. To be removed after 20.0.0.')
SourceTable.getCentroidFlagKey = deprecate_pybind11(
    SourceTable.getCentroidFlagKey,
    reason='Use `getCentroidSlot().getFlagKey()` instead. To be removed after 20.0.0.')
SourceTable.getShapeDefinition = deprecate_pybind11(
    SourceTable.getShapeDefinition,
    reason='Use `getSchema().getAliasMap().get("slot_Shape")` instead. To be removed after 20.0.0.')
SourceTable.hasShapeSlot = deprecate_pybind11(
    SourceTable.hasShapeSlot,
    reason='Use `getShapeSlot().isValid()` instead. To be removed after 20.0.0.')
SourceTable.getShapeKey = deprecate_pybind11(
    SourceTable.getShapeKey,
    reason='Use `getShapeSlot().getMeasKey()` instead. To be removed after 20.0.0.')
SourceTable.getShapeErrKey = deprecate_pybind11(
    SourceTable.getShapeErrKey,
    reason='Use `getShapeSlot().getErrKey()` instead. To be removed after 20.0.0.')
SourceTable.getShapeFlagKey = deprecate_pybind11(
    SourceTable.getShapeFlagKey,
    reason='Use `getShapeSlot().getFlagKey()` instead. To be removed after 20.0.0.')
