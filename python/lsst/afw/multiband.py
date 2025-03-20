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

from deprecated.sphinx import deprecated

from lsst.geom import Box2I


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
    or set the `_bands`, `_singles`, and `_bbox`.

    Parameters
    ----------
    bands: `list`
        List of band names.
    singles: `list`
        List of single band objects
    bbox: `Box2I`
        By default `MultibandBase` uses `singles[0].getBBox()` to set
        the bounding box of the multiband
    """
    def __init__(self, bands, singles, bbox=None):
        self._bands = tuple([f for f in bands])
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
        deep: `bool`
            Whether or not to make a deep copy

        Returns
        -------
        result: `MultibandBase`
            copy of the instance that inherits from `MultibandBase`
        """
        pass

    @property
    @deprecated(reason="This has been replaced with `bands`. Will be removed after v29.",
                version="v29.0", category=FutureWarning)
    def filters(self):
        """List of filter names for the single band objects (deprecated)

        Use `bands` instead.
        """
        return self._bands

    @property
    def bands(self):
        """List of band names for the single band objects
        """
        return self._bands

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
        return len(self.bands)

    def __getitem__(self, args):
        """Get a slice of the underlying array

        If only a single band is specified,
        return the single band object sliced
        appropriately.
        """
        if not isinstance(args, tuple):
            indices = (args,)
        else:
            indices = args

        # Return the single band object if the first
        # index is not a list or slice.
        bands, bandIndex = self._bandNamesToIndex(indices[0])
        if not isinstance(bandIndex, slice) and len(bandIndex) == 1:
            if len(indices) > 2:
                return self.singles[bandIndex[0]][indices[1:]]
            elif len(indices) == 2:
                return self.singles[bandIndex[0]][indices[1]]
            else:
                return self.singles[bandIndex[0]]

        return self._slice(bands=bands, bandIndex=bandIndex, indices=indices[1:])

    def __iter__(self):
        self._bandIndex = 0
        return self

    def __next__(self):
        if self._bandIndex < len(self.bands):
            result = self.singles[self._bandIndex]
            self._bandIndex += 1
        else:
            raise StopIteration
        return result

    def _bandNamesToIndex(self, bandIndex):
        """Convert a list of band names to an index or a slice

        Parameters
        ----------
        bandIndex: iterable or `object`
            Index to specify a band or list of bands,
            usually a string or enum.
            For example `bandIndex` can be
            `"R"` or `["R", "G", "B"]` or `[Band.R, Band.G, Band.B]`,
            if `Band` is an enum.

        Returns
        -------
        bandNames: `list`
            Names of the bands in the slice
        bandIndex: `slice` or `list` of `int`
            Index of each band in `bandNames` in
            `self.bands`.
        """
        if isinstance(bandIndex, slice):
            if bandIndex.start is not None:
                start = self.bands.index(bandIndex.start)
            else:
                start = None
            if bandIndex.stop is not None:
                stop = self.bands.index(bandIndex.stop)
            else:
                stop = None
            bandIndices = slice(start, stop, bandIndex.step)
            bandNames = self.bands[bandIndices]
        else:
            if isinstance(bandIndex, str):
                bandNames = [bandIndex]
                bandIndices = [self.bands.index(bandIndex)]
            else:
                try:
                    # Check to see if the bandIndex is an iterable
                    bandNames = [f for f in bandIndex]
                except TypeError:
                    bandNames = [bandIndex]
                bandIndices = [self.bands.index(f) for f in bandNames]
        return tuple(bandNames), bandIndices

    def setXY0(self, xy0):
        """Shift the bounding box but keep the same Extent

        Parameters
        ----------
        xy0: `Point2I`
            New minimum bounds of the bounding box
        """
        self._bbox = Box2I(xy0, self._bbox.getDimensions())
        for singleObj in self.singles:
            singleObj.setXY0(xy0)

    def shiftedTo(self, xy0):
        """Shift the bounding box but keep the same Extent

        This method is broken until DM-10781 is completed.

        Parameters
        ----------
        xy0: `Point2I`
            New minimum bounds of the bounding box

        Returns
        -------
        result: `MultibandBase`
            A copy of the object, shifted to `xy0`.
        """
        raise NotImplementedError("shiftedBy not implemented until DM-10781")
        result = self.clone(False)
        result._bbox = Box2I(xy0, result._bbox.getDimensions())
        for singleObj in result.singles:
            singleObj.setXY0(xy0)
        return result

    def shiftedBy(self, offset):
        """Shift a bounding box by an offset, but keep the same Extent

        This method is broken until DM-10781 is completed.

        Parameters
        ----------
        offset: `Extent2I`
            Amount to shift the bounding box in x and y.

        Returns
        -------
        result: `MultibandBase`
            A copy of the object, shifted by `offset`
        """
        raise NotImplementedError("shiftedBy not implemented until DM-10781")
        xy0 = self._bbox.getMin() + offset
        return self.shiftedTo(xy0)

    @abstractmethod
    def _slice(self, bands, bandIndex, indices):
        """Slice the current object and return the result

        Different inherited classes will handling slicing differently,
        so this method must be overloaded in inherited classes.

        Parameters
        ----------
        bands: `list` of `str`
            List of band names for the slice. This is a subset of the
            bands in the parent multiband object
        bandIndex: `list` of `int` or `slice`
            Index along the band dimension
        indices: `tuple` of remaining indices
            `MultibandBase.__getitem__` separates the first (band)
            index from the remaining indices, so `indices` is a tuple
            of all of the indices that come after `band` in the
            `args` passed to `MultibandBase.__getitem__`.

        Returns
        -------
        result: `object`
            Sliced version of the current object, which could be the
            same class or a different class depending on the
            slice being made.
        """
        pass

    def __repr__(self):
        result = "<{0}, bands={1}, bbox={2}>".format(
            self.__class__.__name__, self.bands, self.getBBox().__repr__())
        return result

    def __str__(self):
        if hasattr(self, "array"):
            return str(self.array)
        return self.__repr__()
