# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__all__ = ["MultibandBase"]

from abc import ABC, abstractmethod

import numpy as np

from lsst.geom import Point2I, Box2I
from .image import PARENT


class MultibandBase(ABC):
    """Base class for multiband objects

    The LSST stack has a number of image-like classes that have
    data in multiple bands that are stored as separate objects.
    Analyzing the data can be easier using a Multiband object that
    wraps the underlying data as a single data cube that can be sliced and
    updated as a single object.

    `MultibandBase` is designed to contain the most important universal
    methods for initializing, slicing, and extracting common parameters
    (such as the bounding box or XY0 position) to all of the single band classes,
    as long as derived classes either call the base class `__init__`
    or set the `_filters`, `_singles`, and `_bbox`.

    Parameters
    ----------
    filters: list
        List of filter names.
    singles: list
        List of single band objects
    bbox: Box2I
        By default `MultibandBase` uses `singles[0].getBBox()` to set
        the bounding box of the multiband
    """
    def __init__(self, filters, singles, bbox=None):
        self._filters = tuple([f for f in filters])
        self._singles = tuple(singles)        

        if bbox is None:
            self._bbox = self._singles[0].getBBox()
            if not all([s.getBBox() == self.getBBox() for s in self.singles]):
                bboxes = [s.getBBox() == self.getBBox() for s in self.singles]
                err = "`singles` are required to have the same bounding box, received {0}"
                raise ValueError(err.format(bboxes))
        else:
            self._bbox = bbox

    @abstractmethod
    def clone(self, deep=True):
        """Copy the current object

        This must be overloaded in a subclass of `MultibandBase`

        Parameters
        ----------
        deep: bool
            Whether or not to make a deep copy

        Returns
        -------
        result: `MultibandBase`
            copy of the instance that inherits from `MultibandBase`
        """
        pass

    @property
    def filters(self):
        """List of filter names for the single band objects
        """
        return self._filters

    @property
    def singles(self):
        """List of single band objects
        """
        return self._singles

    def getBBox(self):
        """Bounding box
        """
        return self._bbox

    def getXY0(self):
        """Minimum coordinate in the bounding box
        """
        return self.getBBox().getMin()

    @property
    def x0(self):
        """X0

        X component of XY0 `Point2I.getX()`
        """
        return self.getBBox().getMinX()

    @property
    def y0(self):
        """Y0

        Y component of XY0 `Point2I.getY()`
        """
        return self.getBBox().getMinY()

    @property
    def origin(self):
        """Minimum (y,x) position

        This is the position of `self.getBBox().getMin()`,
        but available as a tuple for numpy array indexing.
        """
        return (self.y0, self.x0)

    @property
    def width(self):
        """Width of the images
        """
        return self.getBBox().getWidth()

    @property
    def height(self):
        """Height of the images
        """
        return self.getBBox().getHeight()

    def __len__(self):
        return len(self.filters)

    def __getitem__(self, args):
        """Get a slice of the underlying array

        If only a single filter is specified,
        return the single band object sliced
        appropriately.
        """
        if not isinstance(args, tuple):
            indices = (args,)
        else:
            indices = args

        # Return the single band object if the first
        # index is not a list or slice.
        filters, filterIndex = self._filterNamesToIndex(indices[0])
        if not isinstance(filterIndex, slice) and len(filterIndex) == 1:
            single = self.singles[filterIndex[0]]
            # This temporary code is needed until image-like objects
            # take integer indices, numpy-like slices
            # and Point2I objects
            # Once the image-like API has been updated,
            # this entire "if" block can be replaced with:
            # return self.singles[filterIndex[0]][indices[1:]]
            if len(indices) > 1:
                indexY, indexX = self.imageIndicesToNumpy(indices[1:])
                # Return a scalar if possible
                if isinstance(indices[1], Point2I) or (
                    np.issubdtype(type(indexY), np.integer) and
                    np.issubdtype(type(indexX), np.integer)
                ):
                    return self._slice(filters=filters, filterIndex=filterIndex, indices=indices[1:])
                elif not isinstance(indices[1], Box2I):
                    # Convert indices into a bounding box
                    bbox = self._getBBoxFromIndices((indexY, indexX))
                else:
                    bbox = indices[1]
                result = single.Factory(single, bbox, PARENT)
            else:
                result = single
            return result

        return self._slice(filters=filters, filterIndex=filterIndex, indices=indices[1:])

    def __iter__(self):
        self._filterIndex = 0
        return self

    def __next__(self):
        if self._filterIndex < len(self.filters):
            result = self.singles[self._filterIndex]
            self._filterIndex += 1
        else:
            raise StopIteration
        return result

    def _filterNamesToIndex(self, filterIndex):
        """Convert a list of filter names to an index or a slice

        Parameters
        ----------
        filterIndex: iterable or object
            Index to specify a filter or list of filters,
            usually a string or enum.
            For example `filterIndex` can be
            `"R"` or `["R", "G", "B"]` or `[Filter.R, Filter.G, Filter.B]`,
            if `Filter` is an enum.

        Returns
        -------
        filterNames: list
            Names of the filters in the slice
        filterIndex: slice or list of `int`
            Index of each filter in `filterNames` in
            `self.filters`.
        """
        if isinstance(filterIndex, slice):
            if filterIndex.start is not None:
                start = self.filters.index(filterIndex.start)
            else:
                start = None
            if filterIndex.stop is not None:
                stop = self.filters.index(filterIndex.stop)
            else:
                stop = None
            filterIndices = slice(start, stop, filterIndex.step)
            filterNames = self.filters[filterIndices]
        else:
            try:
                # Check to see if the filterIndex is an iterable
                filterNames = [f for f in filterIndex]
            except TypeError:
                filterNames = [filterIndex]
            filterIndices = [self.filters.index(f) for f in filterNames]
        return tuple(filterNames), filterIndices

    def _indexError(self, indices):
        """To make indexing faster we stringify the error on the fly, if needed
        """
        indexError = "Exected a tuple of 1 or 2 indices, a Point2I, or a Box2I, but received {0}"
        indexError = indexError.format(indices)
        return indexError

    def imageIndicesToNumpy(self, indices):
        """Convert slicing format to numpy

        LSST `afw` image-like objects use an `[x,y]` coordinate
        convention, accept `Point2I` and `Box2I`
        objects for slicing, and slice relative to the
        bounding box `XY0` location;
        while python and numpy use the convention `[y,x]`
        with no `XY0`, so this method converts the `afw`
        indices or slices into numpy indices or slices

        Parameters
        ----------
        indices: `sequence`, `Point2I` or `Box2I`
            An `(xIndex, yIndex)` pair, or a single `(xIndex,)` tuple,
            where `xIndex` and `yIndex` can be a `slice`, `int`,
            or list of `int` objects, and if only a single `xIndex` is
            given, a `Point2I` or `Box2I`.

        Returns
        -------
        sy, sx: tuple of indices or slices
            Indices or slices in python (y,x) ordering, with `XY0` subtracted.
        """
        bbox = self.getBBox()
        x0 = bbox.getMinX()
        y0 = bbox.getMinY()
        sx = None
        sy = None

        # The same IndexError is used in multiple places, so create it once
        # Make sure that `indices` is list-like
        if isinstance(indices, Point2I) or isinstance(indices, Box2I):
            indices = [indices]

        if isinstance(indices[0], Point2I) or isinstance(indices[0], Box2I):
            if len(indices) != 1:
                raise IndexError(self._indexError(indices))
            if isinstance(indices[0], Point2I):
                sx = indices[0].getX() - x0
                sy = indices[0].getY() - y0
            else:
                minSx = indices[0].getMinX() - x0
                minSy = indices[0].getMinY() - y0
                sx = slice(minSx, minSx+indices[0].getWidth())
                sy = slice(minSy, minSy+indices[0].getHeight())
        else:
            if len(indices) == 2:
                sy = indices[1]
                sx = indices[0]
            elif len(indices) == 1:
                sx = indices[0]
            else:
                raise IndexError(self._indexError(indices))

            if sx is not None:
                sx = self._removeOffset(sx, x0, bbox.getMaxX())
            if sy is not None:
                sy = self._removeOffset(sy, y0, bbox.getMaxY())
        return (sy, sx)

    def _removeOffset(self, index, x0, xf):
        """Modify a coordinate or slice by subtracting an offset

        Parameters
        ----------
        index: slice, int, or list of int
            Index in the full image (including `XY0`).
        x0: int
            Minimum value of the x or y coordinate in
            the new boudning box.
        xf: int
            Maximum value of the x or y coordinate in
            the new bounding box.

        Returns
        -------
        newIndex: slice, int, or list of int
            Updated index using `x0`/`y0` and `xf`/`yf`.
        """
        def _applyBBox(index, x0, xf):
            """Throw an error if an invalid index is used
            """
            if index > 0 and (index < x0 or index > xf):
                err = "Indices must be <0 or between {0} and {1}, received {2}"
                raise IndexError(err.format(x0, xf, index))
            newIndex = index - x0
            if index < 0:
                newIndex += xf
            return newIndex

        if index is None:
            return index

        newIndex = None
        if isinstance(index, slice):
            if index.start is None:
                start = None
            else:
                start = _applyBBox(index.start, x0, xf)
            if index.stop is None:
                stop = None
            else:
                stop = _applyBBox(index.stop, x0, xf)
            if index.step is not None and index.step != 1:
                raise IndexError("Image slicing must be contiguous")
            newIndex = slice(start, stop)
        elif hasattr(index, "__len__"):
            newIndex = [_applyBBox(i, x0, xf) for i in index]
        else:
            newIndex = _applyBBox(index, x0, xf)
        return newIndex

    def shiftedTo(self, xy0):
        """Shift the bounding box but keep the same Extent

        Parameters
        ----------
        xy0: `Point2I`
            New minimum bounds of the bounding box
        """
        self._bbox = Box2I(xy0, self._bbox.getDimensions())
        for singleObj in self.singles:
            singleObj.setXY0(xy0)

    def shiftedBy(self, offset):
        """Shift a bounding box by an offset, but keep the same Extent

        Parameters
        ----------
        offset: `Extent2I`
            Amount to shift the bounding box in x and y.
        """
        xy0 = self._bbox.getMin() + offset
        self.shiftedTo(xy0)

    def _getBBoxFromIndices(self, indices):
        """Set the current bounding box from a set of slices

        This method creates the bounding box for this
        object based on the bounding box of it's parent,
        and sets the `self_bbox` parameter, accessible
        by `self.getBBox()`.

        Parameters
        ----------
        indices: tuple of indices or slices
            Slices in x and y from the parent array.

        Returns
        -------
        bbox: `Box2I`
            Bounding box for the image portion of the multiband object.
        """
        oldBBox = self.getBBox()
        yx0 = [oldBBox.getMinY(), oldBBox.getMinX()]
        yxF = [oldBBox.getMaxY(), oldBBox.getMaxX()]

        for idx, index in enumerate(indices):
            if isinstance(index, slice):
                _start = yx0[idx]
                if index.start is not None:
                    yx0[idx] += index.start
                if index.stop is not None:
                    yxF[idx] = _start + index.stop - 1
            else:
                if hasattr(index, "__len__"):
                    raise IndexError("MultibandBase objects must have contiguous slices")
                else:
                    if index is not None:
                        yx0[idx] += index
                        yxF[idx] = yxF[idx] + 1
        xy0 = Point2I(yx0[1], yx0[0])
        xyF = Point2I(yxF[1], yxF[0])
        bbox = Box2I(xy0, xyF)
        return bbox

    @abstractmethod
    def _slice(self, filters, filterIndex, indices):
        """Slice the current object and return the result

        Different inherited classes will handling slicing differently,
        so this method must be overloaded in inherited classes.

        Parameters
        ----------
        filters: list of str
            List of filter names for the slice. This is a subset of the
            filters in the parent multiband object
        filterIndex: list of indices or slice
            Index along the filter dimension
        indices: tuple of remaining indices
            `MultibandBase.__getitem__` separates the first (filter)
            index from the remaining indices, so `indices` is a tuple
            of all of the indices that come after `filter` in the
            `args` passed to `MultibandBase.__getitem__`.

        Returns
        -------
        result: object
            Sliced version of the current object, which could be the
            same class or a different class depending on the
            slice being made.
        """
        pass

    def __repr__(self):
        result = "<{0}, filters={1}, bbox={2}>".format(
            self.__class__.__name__, self.filters, self.getBBox().__repr__())
        return result

    def __str__(self):
        if hasattr(self, "array"):
            return str(self.array)
        return self.__repr__()
