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
from ._base import Catalog
from ._table import SourceCatalog, SourceTable

Catalog.register("Source", SourceCatalog)


@continueClass
class SourceCatalog:

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
        parent : `int` or `iterable` of `int`
            ID(s) of the parent(s) to get children for.
        args : `~lsst.afw.table.Catalog`
            Additional catalogs to subset for the children to return.

        Returns
        -------
        children : a single iterable of `~lsst.afw.table.SourceRecord`
            Children sources if ``parent`` is of type `int`, or a generator
            yielding a `~lsst.afw.table.SourceRecord`s Children sources for
            each parent if ``parent`` is an `iterable`.

        Raises
        ------
        AssertionError
            Raised if the catalog is not sorted by the parent key.

        Notes
        -----
        Each call to this function checks if the catalog is sorted, which is
        of O(n) complexity, while fetching the children is of O(log n). To
        minimize the computational overhead, it is preferable to prepare an
        iterable of parent ids for which the children need to be fetched and
        pass the iterable as ``parent``.
        """
        if not self.isSorted(SourceTable.getParentKey()):
            raise AssertionError(
                "The table is not sorted by parent, so cannot getChildren")

        def _getChildrenWithoutChecking(parent):
            """Return the subset of self for which the parent field equals the
            given value.

            This function works as desired only if `self` is sorted by the
            parent key, but does not check if it is sorted. This function must
            be used only after ensuring outside of the function that
            self.isSorted(SourceTable.getParentKey() evaluates to True.

            Parameter
            ---------
            parent : `int`
                ID of the parent to get children for.

            Returns
            -------
            children : iterable of `~lsst.afw.table.SourceRecord`
                Children sources.
            """
            s = self.equal_range(parent, SourceTable.getParentKey())
            if args:
                return (self[s],) + tuple(arg[s] for arg in args)
            else:
                return self[s]

        try:
            return (_getChildrenWithoutChecking(p) for p in parent)
        except TypeError:
            return _getChildrenWithoutChecking(parent)
