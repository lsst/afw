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

from lsst.geom import Point2I, Box2I, Extent2I
from .image import PARENT, LOCAL
from .image.slicing import interpretSliceArgs


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

    def getBBox(self, origin=PARENT):
        """Bounding box
        """
        if origin == PARENT:
            return self._bbox
        elif origin == LOCAL:
            return Box2I(Point2I(0, 0), self._bbox.getDimensions())
        raise ValueError("Unrecognized origin, expected either PARENT or LOCAL")

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
            if len(indices) > 2:
                return self.singles[filterIndex[0]][indices[1:]]
            elif len(indices) == 2:
                return self.singles[filterIndex[0]][indices[1]]
            else:
                return self.singles[filterIndex[0]]

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

    def imageIndicesToNumpy(self, sliceArgs):
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
        sliceArgs: `sequence`, `Point2I` or `Box2I`
            An `(xIndex, yIndex)` pair, or a single `(xIndex,)` tuple,
            where `xIndex` and `yIndex` can be a `slice` or `int`,
            or list of `int` objects, and if only a single `xIndex` is
            given, a `Point2I` or `Box2I`.

        Returns
        -------
        y: index or slice
            Index or slice in the y dimension
        x: index or slice
            Index or slice in the x dimension
        bbox: Box2I
            Bounding box of the image.
            If `bbox` is `None` then the result is a point and
            not a subset of an image.
        """
        # Use a common slicing algorithm as single band images
        x, y, origin = interpretSliceArgs(sliceArgs, self.getBBox)

        if origin == PARENT:
            if isinstance(x, slice):
                assert isinstance(y, slice)
                bbox = Box2I(Point2I(x.start, y.start), Extent2I(x.stop-x.start, y.stop-y.start))
                x = slice(x.start - self.x0, x.stop - self.x0)
                y = slice(y.start - self.y0, y.stop - self.y0)
            else:
                x = x - self.x0
                y = y - self.y0
                bbox = None
            return y, x, bbox
        elif origin != LOCAL:
            raise ValueError("Unrecognized value for origin")

        # Use a local bounding box
        if isinstance(x, slice):
            assert isinstance(y, slice)
            bbox = Box2I(Point2I(x.start + self.x0, y.start + self.y0),
                         Extent2I(x.stop-x.start, y.stop-y.start))
        else:
            bbox = None
        return y, x, bbox

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
