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

__all__ = ["MultibandPixel", "MultibandImage", "MultibandMask", "MultibandMaskedImage", "MultibandExposure"]

import numpy as np

from lsst.geom import Point2I, Box2I, Extent2I
from . import ImageF, MaskedImageF, Mask, ExposureF
from . import maskedImage as afwMaskedImage
from lsst.afw.multiband import MultibandBase


class MultibandPixel(MultibandBase):
    """Multiband Pixel class

    This represent acts as a container for a single pixel
    (scaler) in multiple bands.
    There are a few methods from `MultibandBase` that are overloaded,
    since a `MultibandPixel` has a `Point2I` for it's boudning box
    as opposed to a `Box2I`.
    """
    def __init__(self, filters, singles, bbox):
        """Initialize a `MultibandPixel` object

        Parameters
        ----------
        singles: list or `MultibandPixel`
            Either a list of single band objects or
            an instance of a `MultibandPixel`
        filters: list
            List of filter names. If `singles` is an `OrderedDict` or
            a `MultibandPixel` then this arguement is ignored,
            otherwise it is required.
        bbox: Point2I
            Location of the pixel in the parent image.
            Unlike other objects that inherit from `MultibandBase`,
            `MultibandPixel` objects don't have a full `Box2I`
            bounding box, since they only contain a single pixel,
            so the bounding box cannot be inherited from the
            list of `singles`.
        """
        if any([arg is None for arg in [singles, filters, bbox]]):
            err = "Expected an array of `singles`, a list of `filters, and a `bbox`"
            raise NotImplementedError(err)

        self._singles = np.array(singles)
        self._filters = tuple(filters)
        self._bbox = bbox
        self._singleType = self.singles.dtype

        # Make sure that the bounding box has been setup properly
        if not isinstance(self.getBBox(), Point2I):
            err = ("Something went wrong, the bounding box for a `MultibandPixel` "
                   "should always be a `Point2I`, received {0}")
            raise RuntimeError(err.format(self.getBBox()))

    def _getArray(self):
        """Data cube array in multiple bands

        Since `self._singles` is just a 1D array,
        `array` just returns `self._singles`.
        """
        return self.singles

    def _setArray(self, value):
        assert value.shape == self.array.shape
        self._singles = value

    array = property(_getArray, _setArray)

    def copy(self, deep=False):
        """Make a copy of the current instance

        `MultibandPixel._singles` is an array,
        so this just makes a copy of the array
        (as opposed to a view of the parent array).
        """
        if deep:
            filters = tuple([f for f in self.filters])
            singles = np.copy(self.singles)
            # For MultibandPixels, `bbox` is a `Point2I`
            coords = self.getBBox()
            bbox = Point2I(coords.getX(), coords.getY())
        else:
            filters = self.filters
            singles = self.singles
            bbox = self.getBBox()
        return MultibandPixel(filters=filters, singles=singles, bbox=bbox)

    def __getitem__(self, indices):
        """Get a slice of the underlying array

        Since a `MultibandPixel` is a scalar in the
        spatial dimensions, it can only be indexed with
        a filter name, number, or slice.
        """
        if isinstance(indices, tuple):
            err = ("Too many indices: "
                   "`MultibandPixel has no spatial dimensions and "
                   "only accepts a filterIndex")
            raise IndexError(err)
        # Make the index into a valid numpy index or slice
        filters, filterIndex = self.filterToIndex(indices)
        if len(filters) == 1 and not isinstance(filterIndex, slice):
            # The user only requested a pixel in a single band, so return it
            return self.array[filterIndex[0]]

        result = self.copy()
        result._filters = filters
        result._singles = self._singles[filterIndex]
        # No need to update the bounding box, since pixels can only be sliced in the filter dimension
        return result

    def setBBox(self, bbox):
        """Overload `MultibandBase.setBBox`

        Since the "single band objects" are elements
        of an array and do not have a bounding box,
        overloading `setBBox` is needed to prevent the
        code from attempting to set the bounding box of
        each item in `singles`.

        Parameters
        ----------
        bbox: Point2I
            Pixel location in the parent image
        """
        if not isinstance(bbox, Point2I):
            err = "The bounding box for a `MultibandPixel` should always be a `Point2I`, received {0}"
            raise ValueError(err.format(bbox))
        self._bbox = bbox


class MultibandImage(MultibandBase):
    """Multiband Image class

    This class acts as a container for multiple `afw.Image` objects.
    All images must be contained in the same bounding box,
    and have the same data type.
    """
    def __init__(self, filters, singles=None, array=None, bbox=None,
                 singleType=ImageF, filterKwargs=None, **kwargs):
        """Initialize a `MultibandBase` object

        Parameters
        ----------
        singles: list
            A list of single band objects.
            If `array` is not `None`, then `singles` is ignored
        filters: list
            List of filter names. If `singles` is an `OrderedDict`
            then this argument is ignored, otherwise it is required.
        array: 3D numpy array
            Array (filters, y, x) of multiband data.
            If this is used to initialize a `MultibandImage`,
            either `bbox` or `singles` is also required.
        singleType: class
            Class of the single band objects.
            This is ignored unless `singles` and `array`
            are both `None`, in which case it is required.
        bbox: `Box2I`
            Location of the array in the parent image.
            This argument is ignored if `singles` is not `None`.
        filterKwargs: dict
            Keyword arguments to pass to `_fullInitialize` to
            initialize a new instance of an inherited class
            that are different for each filter.
            The keys are the names of the arguments and the values
            should also be dictionaries, with filter names as keys
            and the value of the argument for a given filter as values.
        kwargs: dict
            Keyword arguments to pass to `_fullInitialize` to
            initialize a new instance of an inherited class that are the
            same in all bands.
        """
        # Create the single band objects from the array
        if array is not None and filters is not None:
            if len(array) != len(filters):
                raise ValueError("`array` and `filters` must have the same length")
            self._array = array
            self._filters = filters
            if singles is None:
                if bbox is None:
                    bbox = Box2I(Point2I(0, 0), Extent2I(array[0].shape[1], array[0].shape[0]))
                self._bbox = bbox
                self._updateSingles(singleType)
            else:
                if len(singles) != len(array):
                    raise ValueError("`array` should have the same length as `singles`")
                if not np.all([single.array.shape == array.shape[1:] for single in singles]):
                    err = ("`array` and each item in `singles` must have the same spatial dimension, "
                           "received dimensions {0} for array with dimension {1}")
                    raise ValueError(err.format([single.array.shape for single in singles], array.shape))
                self._singles = tuple(singles)
                self._bbox = singles[0].getBBox()
        else:
            # Extract the single band objects and filters
            if singles is None and filters is not None:
                # Attempt to load a set of images
                self._singleType = singleType
                singles = []
                for f in filters:
                    if filterKwargs is not None:
                        for key, value in filterKwargs:
                            kwargs[key] = value[f]
                    singles.append(singleType(**kwargs))
            if singles is not None:
                super().__init__(filters, singles)
            else:
                err = ("Either a list of `singles` and `filters`; "
                       "or an `array`, `filters`, `bbox`, and `singleType`; "
                       "or a set of `kwargs` is required to initialize a `MultibandImage`.")
                raise NotImplementedError(err)

            self._array = np.array([image.array for image in self.singles])
            self._updateSingles(type(self.singles[0]))

        # Make sure that all of the parameters have been setup appropriately
        if not isinstance(self._bbox, Box2I):
            raise RuntimeError("Something went wrong, `self._bbox` should be a `Box2I`")
        if not np.all([single.getBBox() == self.getBBox() for single in self.singles]):
            raise ValueError("Single band images did not all have the same bounding box")

    def _getArray(self):
        """Data cube array in multiple bands
        """
        return self._array

    def _setArray(self, value):
        self.array[:] = value

    array = property(_getArray, _setArray)

    def _updateSingles(self, singleType):
        """Update the Image<X> in each band

        This method is called when a `MultibandImage` is initialized.
        """
        self._singleType = singleType
        self._singles = self._arrayToSingles(self.array, singleType, self.getXY0())
        if not len(self.filters) == len(self.singles):
            raise RuntimeError("The length of `filters` and `array` should be the same")

    def _arrayToSingles(self, array, singleType, xy0):
        """Create a list of `Image<X>` objects from a 3D array
        """
        return tuple([singleType(array=array[n], xy0=xy0) for n in range(len(array))])

    def copy(self, deep=False):
        """Copy the current object

        Parameters
        ----------
        deep: bool
            Whether or not to make a deep copy
        """
        if deep:
            filters = tuple([f for f in self.filters])
            array = np.copy(self.array)
            bbox = Box2I(self.getBBox())
            result = type(self)(filters=filters, array=array, bbox=bbox, singleType=self.singleType)
        else:
            result = type(self)(self.filters, self.singles, array=self.array)
        return result

    def _slice(self, filters, filterIndex, indices):
        """Slice the current object and return the result

        See `Multiband._slice` for a list of the parameters.
        """
        if len(indices) > 0:
            allSlices = [filterIndex, slice(None), slice(None)]
            sy, sx = self.imageIndicesToNumpy(indices)
            if sy is not None:
                allSlices[-2] = sy
            if sx is not None:
                allSlices[-1] = sx
            array = self._array[allSlices]

            # Return a scalar or MultibandPixel
            # if the image indices are integers
            if np.issubdtype(type(sy), np.integer) and np.issubdtype(type(sx), np.integer):
                if not isinstance(filterIndex, slice) and len(filterIndex) == 1:
                    return array[0]
                result = MultibandPixel(
                    singles=array,
                    filters=filters,
                    bbox=Point2I(sx + self.x0, sy + self.y0)
                )
                return result
            # Set the bbox size based on the slices
            bbox = self.getBBoxFromIndices(allSlices[1:])
            singles = self._arrayToSingles(array, self.singleType, bbox.getMin())
            result = type(self)(filters, singles, array=array)
        else:
            result = type(self)(filters=filters, array=self._array[filterIndex], bbox=self.getBBox())

        # Check that the image and array shapes agree
        imageShape = (
            len(result.filters),
            result.getBBox().getHeight(),
            result.getBBox().getWidth()
        )
        if result.array.shape != imageShape:
            err = ("Something went wrong with the internal slicing mechanism, "
                   "the array shape {0} != the image shape {1}.")
            raise RuntimeError(err.format(result.array.shape, imageShape))
        return result


class MultibandMask(MultibandImage):
    def __init__(self, filters, singles=None, array=None, bbox=None,
                 singleType=Mask, filterKwargs=None, **kwargs):
        """Initialize a `MultibandMask` object

        See `MultibandImage` for a description of the parameters..
        """
        super().__init__(filters, singles, array, bbox, singleType, filterKwargs, **kwargs)
        # Set Mask specific properties
        refMask = self._singles[0]
        self._maskPlaneDict = refMask.getMaskPlaneDict()
        assert np.all([refMask.getMaskPlaneDict() == m.getMaskPlaneDict() for m in self.singles])
        self._getNumPlanesMax = refMask.getNumPlanesMax()
        assert np.all([refMask.getNumPlanesMax() == m.getNumPlanesMax() for m in self.singles])
        self._getNumPlanesUsed = refMask.getNumPlanesUsed()
        assert np.all([refMask.getNumPlanesUsed() == m.getNumPlanesUsed() for m in self.singles])

    def getMaskPlane(self, key):
        return self.getMaskPlaneDict()[key]

    def getNumPlanesMax(self):
        """Maximum number of mask planes available

        This is required to be the same for all of the single
        band `Mask` objects.
        """
        return self._getNumPlanesMax

    def getNumPlanesUsed(self):
        """Number of mask planes used

        This is required to be the same for all of the single
        band `Mask` objects.
        """
        return self._getNumPlanesUsed

    def getMaskPlaneDict(self):
        """Dictionary of Mask Plane bit values
        """
        return self._maskPlaneDict

    def clearMaskPlaneDict(self):
        """Reset the mask plane dictionary
        """
        mask = self._singles[0]
        mask.clearMaskPlaneDict()
        self._maskPlaneDict = mask.getMaskPlaneDict()

    def addMaskPlane(self, name):
        """Add a mask to the mask plane

        Parameters
        ----------
        name: str
            Name of the new mask plane
        """
        mask = self._singles[0]
        idx = mask.addMaskPlane(name)
        self._maskPlaneDict = mask.getMaskPlaneDict()
        return idx

    def removeMaskPlane(self, name):
        """Remove a mask from the mask plane

        Parameters
        ----------
        name: str
            Name of the mask plane to remove
        """
        mask = self._singles[0]
        mask.removeMaskPlane(name)
        self._maskPlaneDict = mask.getMaskPlaneDict()

    def clearAllMaskPlanes(self):
        """Clear all the pixels
        """
        mask = self._singles[0]
        mask.clearAllMaskPlanes()

    def _getOtherMasks(self, others):
        """Check if two masks can be combined

        This method checks that `self` and `others`
        have the same number of bands, or if
        others is a single value, creates a list
        to use for updating all of the `singles`.
        """
        if isinstance(others, MultibandMask):
            if len(self.singles) != len(others.singles) or self.filters != others.filters:
                msg = "Both `MultibandMask` objects must have the same number of bands to combine"
                raise ValueError(msg)
            result = [s for s in others.singles]
        else:
            result = [others]*len(self.singles)
        return result

    def __ior__(self, others):
        _others = self._getOtherMasks(others)
        singles = list(self.singles)
        for n in range(len(self)):
            singles[n] |= _others[n]
        self._singles = tuple(singles)
        return self

    def __iand__(self, others):
        _others = self._getOtherMasks(others)
        singles = list(self.singles)
        for n in range(len(self)):
            singles[n] &= _others[n]
        self._singles = tuple(singles)
        return self

    def __ixor__(self, others):
        _others = self._getOtherMasks(others)
        singles = list(self.singles)
        for n in range(len(self)):
            singles[n] ^= _others[n]
        self._singles = tuple(singles)
        return self

    def __setitem__(self, args, value):
        """Set a subset of the MultibandMask
        """
        if not isinstance(args, tuple):
            indices = (args,)
        else:
            indices = args

        # Return the single band object if the first
        # index is not a list or slice.
        filters, filterIndex = self.filterToIndex(indices[0])
        if len(indices) > 1:
            sy, sx = self.imageIndicesToNumpy(indices[1:])
        else:
            sy = sx = slice(None)
        if isinstance(value, Mask):
            self._array[filterIndex, sy, sx] = value.array
        else:
            self._array[filterIndex, sy, sx] = value

    def set(self, value, filters=None, x=None, y=None):
        """Set the value of the mask

        Parameters
        ----------
        filters: str name, index, or slice
            Filter(s) to set to `value`.
            If filters is `None` then all filters are used.
        value: int or array
            Value(s) to set for the assigned filter(s)
        x: int
            Optional x-position of a pixel to set
        y: int
            Optional y-position of a pixel to set
        """
        if (x is None) ^ (y is None):
            err = "Must specify either `x` and `y` coordinate or no coordinates"
            raise IndexError(err)
        if x is not None:
            if not np.issubdtype(type(y), np.integer) or not np.issubdtype(type(x), np.integer):
                raise IndexError("x and y must be integers if they are specified")
            # Temporarily adjust for XY0, since Mask does not support XY0 yet
            x -= self.x0
            y -= self.y0

        if filters is not None:
            filters, filterIndex = self.filterToIndex(filters)
        else:
            filterIndex = range(len(self))
        if isinstance(filterIndex, slice):
            filterIndex = np.arange(len(self))[filterIndex]

        if hasattr(value, "__len__"):
            for f in filterIndex:
                if x is None:
                    self.singles[f].set(value[f])
                else:
                    self.singles[f].set(x, y, value[f])
        else:
            for f in filterIndex:
                if x is None:
                    self.singles[f].set(value)
                else:
                    self.singles[f].set(x, y, value)


class MultibandTripleBase(MultibandBase):
    """MultibandTripleBase class

    This is a base class inherited by multiband classes
    with `image`, `mask`, and `variance` objects,
    such as `MultibandMaskedImage` and `MultibandExposure`.
    """
    def __init__(self, filters, singles=None, image=None, mask=None, variance=None,
                 singleType=MaskedImageF, filterKwargs=None, **kwargs):
        """Initialize a `MultibandTripleBase` object

        Parameters
        ----------
        singles: list or `OrderedDict`
            Either a list of single band objects or an `OrderedDict` with
            filter names as keys and single band objects as values.
            If no `singles` are specified then `image`, `mask`, and `variance`
            must be given.
        filters: list
            List of filter names. If `singles` is an `OrderedDict`
            then this argument is ignored, otherwise it is required.
        image: list
            List of `Image` objects that represent the image in each band.
            Ignored if `singles` is not `None`.
        mask: list
            List of `Mask` objects that represent the mask in each band.
            Ignored if `singles` is not `None`.
        variance: list
            List of `Image` objects that represent the variance in each band.
            Ignored if `singles` is not `None`.
        singleType: class
            Class of single band objects.
        filterKwargs: dict
            Keyword arguments to pass to `_fullInitialize` to
            initialize a new instance of an inherited class
            that are different for each filter.
            The keys are the names of the arguments and the values
            should also be dictionaries, with filter names as keys
            and the value of the argument for a given filter as values.
        kwargs: dict
            Keyword arguments to pass to `_fullInitialize` to
            initialize a new instance of an inherited class that are the
            same in all bands.
        """
        if singles is not None:
            # Extract the single band objects and filters
            super().__init__(filters, singles)
            image = [s.image for s in self._singles]
            mask = [s.mask for s in self._singles]
            variance = [s.variance for s in self._singles]
            self._setMultiband(image, mask, variance, filters)
        elif image is not None or mask is not None or variance is not None:
            if image is None or mask is None or variance is None or filters is None:
                err = ("`MultibandTripleBase` must be initialized with "
                       "`singles` and `filters` or `image`, `mask`, `variance`, and `filters`")
                raise ValueError(err)
            isMultiband = [isinstance(m, MultibandBase) for m in [image, mask, variance]]
            if np.any(isMultiband):
                if not np.all(isMultiband):
                    err = "`image`, `mask`, `variance` must all be either multiband or single band"
                    raise ValueError(err)
                self._image = image
                self._mask = mask
                self._variance = variance
            else:
                self._setMultiband(image, mask, variance, filters)
            self._filters = tuple(filters)
            self._singleType = singleType
        elif filters is not None:
            # Extract the single band objects and filters
            self.singleType = singleType
            singles = []
            for f in self.filters:
                if filterKwargs is not None:
                    for key, value in filterKwargs:
                        kwargs[key] = value[f]
                singles.append(self.singleType(**kwargs))
            super().__init__(filters, singles)
        else:
            err = ("Either a list of `singles` and `filters`; "
                   "a set of `filters`, `image`, `mask`, and `variance`; "
                   "or a set of `kwargs` is required to initialize a `MultibandImage`.")
            raise NotImplementedError(err)

        self._singles = self._buildSingles(self._image, self._mask, self._variance)

    def _setMultiband(self, image, mask, variance, filters):
        """Set image, mask, and variance to the multiband objects

        See `MultibandTripleBase` for parameter descriptions.
        """
        self._image = MultibandImage(filters, image)
        self._mask = MultibandMask(filters, mask)
        self._variance = MultibandImage(filters, variance)

    def setBBox(self, bbox):
        """Set the bounding box

        This is different than `MultibandBase.setBBox`
        because the multiband `image`, `mask`, and `variance` objects
        must all have their bounding boxes updated.

        Parameters
        ----------
        bbox: `Box2I` or tuple of indices
            Bounding box to set as the current bounding box.
            If a tuple of indices is passed, a new bounding box
            is created from them and used.
        """
        super().setBBox(bbox)
        self.image.setBBox(bbox)
        self.mask.setBBox(bbox)
        self.variance.setBBox(bbox)

    def copy(self, deep=False):
        """Make a copy of the current instance
        """
        if deep:
            filters = tuple([f for f in self.filters])
            singles = tuple([self.singleType(s, deep=True) for s in self.singles])
            result = type(self)(filters=filters, singles=singles)
        else:
            result = type(self)(image=self.image, mask=self.mask, variance=self.variance,
                                filters=self.filters)
        return result

    def _slice(self, filters, filterIndex, indices):
        """Slice the current object and return the result

        See `Multiband._slice` for a list of the parameters.
        """
        image = self.image._slice(filters, filterIndex, indices)
        mask = self._mask._slice(filters, filterIndex, indices)
        variance = self._variance._slice(filters, filterIndex, indices)

        result = type(self)(filters=filters, image=image, mask=mask, variance=variance)
        assert np.all([r.getBBox() == result._bbox for r in [result._mask, result._variance]])
        return result

    def _verifyUpdate(self, image=None, mask=None, variance=None):
        """Check that the new image, mask, or variance is valid

        This basically means checking that the update to the
        property matches the current bounding box and inherits
        from the `MultibandBase` class.
        """
        for prop in [image, mask, variance]:
            if prop is not None:
                if prop.getBBox() != self.getBBox():
                    raise ValueError("Bounding box does not match the current class")
                if not isinstance(prop, MultibandBase):
                    err = "image, mask, and variance should all inherit from the MultibandBase class"
                    raise ValueError(err)

    def getImage(self):
        """Get the image
        """
        return self._image

    def setImage(self, image):
        """Set the image
        """
        self._verifyUpdate(image=image)
        self._image = image
        self._singles = self._buildSingles(image=image)

    image = property(getImage, setImage)

    def getMask(self):
        """Get the mask
        """
        return self._mask

    def setMask(self, mask):
        """Set the mask
        """
        self._verifyUpdate(mask=mask)
        self._mask = mask
        self._singles = self._buildSingles(mask=mask)

    mask = property(getMask, setMask)

    def getVariance(self):
        """Get the variance
        """
        return self._variance

    def setVariance(self, variance):
        """Set the variance
        """
        self._verifyUpdate(variance=variance)
        self._variance = variance
        self._singles = self._buildSingles(variance=variance)

    variance = property(getVariance, setVariance)


class MaskedPixel(object):
    """A single pixel with an image, mask, and variance
    """
    def __init__(self, image, mask, variance, bbox):
        self._image = image
        self._mask = mask
        self._variance = variance
        self._bbox = bbox

        assert isinstance(self.getBBox(), Point2I)

    def copy(self, deep=False):
        """Make a copy of the current instance

        `image`, `mask`, and `variance` are all just
        numbers, so only the bounding boxes changes
        for deep copies.
        """
        if deep:
            bbox = Point2I(self.getBBox())
        else:
            bbox = self.getBBox()
        return MaskedPixel(self.image, self.mask, self.variance, bbox)

    def getImage(self):
        return self._image

    def setImage(self, value):
        assert np.isscalar(value)
        self._image = value

    image = property(getImage, setImage)

    def getMask(self):
        return self._mask

    def setMask(self, value):
        assert np.issubdtype(type(value), np.integer)
        self._mask = value

    mask = property(getMask, setMask)

    def getVariance(self):
        return self._variance

    def setVariance(self, value):
        assert np.isscalar(value)
        self._variance = value

    variance = property(getVariance, setVariance)

    def getBBox(self):
        return self._bbox


class MultibandMaskedPixel(MultibandTripleBase):
    """MultibandMaskedPixel class

    This class acts as a container for multiple `MaskedPixel` objects.
    All masked pixels must have the same bounding box (`Point2I`),
    and the associated pixels in each band must all have the same
    data types for the `image`, `mask`, and `variance` objects.
    """
    def __init__(self, filters, singles=None, image=None, mask=None, variance=None, bbox=None):
        """Initialize a `MultibandMaskedPixel` object

        See `MultibandTripleBase` for parameter definitions.
        """
        self._bbox = bbox
        super().__init__(filters, singles, image, mask, variance, MaskedPixel)
        # Make sure that the bounding box has been setup properly
        if not isinstance(self.getBBox(), Point2I):
            err = ("Something went wrong, the bounding box for a `MultibandPixel` "
                   "should always be a `Point2I`, received {0}")
            raise RuntimeError(err.format(self.getBBox()))

    def _buildSingles(self, image=None, mask=None, variance=None):
        """Make a new list of single band objects

        Parameters
        ----------
        image: list
            List of `Image` objects that represent the image in each band.
        mask: list
            List of `Mask` objects that represent the mask in each band.
        variance: list
            List of `Image` objects that represent the variance in each band.

        Returns
        -------
        singles: list
            List of `MaskedImage` objects for each band,
            where the `image`, `mask`, and `variance` of each `single`
            point to the multiband objects.
        """
        singles = []
        if image is None:
            image = self.image
        if mask is None:
            mask = self.mask
        if variance is None:
            variance = self.variance

        for n in range(len(image)):
            single = self.singleType(image=image[n], mask=mask[n], variance=variance[n], bbox=self.getBBox())
            singles.append(single)
        return tuple(singles)

    def _setMultiband(self, image, mask, variance, filters):
        """Set image, mask, and variance to the multiband objects

        See `MultibandTripleBase` for parameter descriptions.
        """
        self._image = MultibandPixel(filters, image, bbox=self.getBBox())
        self._mask = MultibandPixel(filters, mask, bbox=self.getBBox())
        self._variance = MultibandPixel(filters, variance, bbox=self.getBBox())

    def copy(self, deep=False):
        """Make a copy of the current instance

        `MultibandPixel._singles` is an array,
        so this just makes a copy of the array
        (as opposed to a view of the parent array).
        """
        if deep:
            filters = tuple([f for f in self.filters])
            singles = tuple([s.copy(True) for s in self.singles])
            # For MultibandPixels, `bbox` is a `Point2I`
            coords = self.getBBox()
            bbox = Point2I(coords.getX(), coords.getY())
        else:
            filters = self.filters
            singles = self.singles
            bbox = self.getBBox()
        return MultibandMaskedPixel(filters=filters, singles=singles, bbox=bbox)

    def __getitem__(self, indices):
        """Get a slice of the underlying array

        Since a `MultibandPixel` is a scalar in the
        spatial dimensions, it can only be indexed with
        a filter name, number, or slice.
        """
        image = self.image[indices]
        mask = self.mask[indices]
        variance = self.variance[indices]

        if hasattr(image, "__len__"):
            result = MultibandMaskedPixel(filters=self.filters, image=image, mask=mask,
                                          variance=variance, bbox=self.getBBox())
        else:
            result = MaskedPixel(image, mask, variance, self.getBBox())
        return result

    def setBBox(self, bbox):
        """Overload `MultibandBase.setBBox`

        Since the "single band objects" are elements
        of an array and do not have a bounding box,
        overloading `setBBox` is needed to prevent the
        code from attempting to set the bounding box of
        each item in `singles`.

        Parameters
        ----------
        bbox: Point2I
            Pixel location in the parent image
        """
        if not isinstance(bbox, Point2I):
            err = "The bounding box for a `MultibandPixel` should always be a `Point2I`, received {0}"
            raise ValueError(err.format(bbox))
        self._bbox = bbox


class MultibandMaskedImage(MultibandTripleBase):
    """MultibandMaskedImage class

    This class acts as a container for multiple `afw.MaskedImage` objects.
    All masked images must have the same bounding box, and the associated
    images must all have the same data type.
    """
    def __init__(self, filters, singles=None, image=None, mask=None, variance=None,
                 filterKwargs=None, **kwargs):
        """Initialize a `MultibandMaskedImage` object

        See `MultibandTripleBase` for parameter definitions.
        """
        super().__init__(filters, singles, image, mask, variance, MaskedImageF, filterKwargs, **kwargs)
        self._bbox = self.singles[0].getBBox()
        if not np.all([single.getBBox() == self.getBBox() for single in self.singles]):
            raise ValueError("Single band masked images did not all have the same bounding box")

    def _buildSingles(self, image=None, mask=None, variance=None):
        """Make a new list of single band objects

        Parameters
        ----------
        image: list
            List of `Image` objects that represent the image in each band.
        mask: list
            List of `Mask` objects that represent the mask in each band.
        variance: list
            List of `Image` objects that represent the variance in each band.

        Returns
        -------
        singles: list
            List of `MaskedImage` objects for each band,
            where the `image`, `mask`, and `variance` of each `single`
            point to the multiband objects.
        """
        singles = []
        if image is None:
            image = self.image
        if mask is None:
            mask = self.mask
        if variance is None:
            variance = self.variance

        for n in range(len(image)):
            single = self.singleType(image=image[n], mask=mask[n], variance=variance[n])
            singles.append(single)
        return tuple(singles)


class MultibandExposure(MultibandTripleBase):
    """MultibandExposure class

    This class acts as a container for multiple `afw.Exposure` objects.
    All exposures must have the same bounding box, and the associated
    images must all have the same data type.
    """
    def __init__(self, filters, singles=None, image=None, mask=None, variance=None,
                 psfs=None, filterKwargs=None, singleType=ExposureF, **kwargs):
        """Initialize a `MultibandMaskedImage` object

        See `MultibandTripleBase` for parameter definitions.
        """
        super().__init__(filters, singles, image, mask, variance, singleType, filterKwargs, **kwargs)
        if psfs is not None:
            self.setAllPsfs(psfs)
        self._bbox = self.singles[0].getBBox()
        if not np.all([single.getBBox() == self.getBBox() for single in self.singles]):
            raise ValueError("Single band masked images did not all have the same bounding box")
        self._psfImage = None

    def _buildSingles(self, image=None, mask=None, variance=None):
        """Make a new list of single band objects

        Parameters
        ----------
        image: list
            List of `Image` objects that represent the image in each band.
        mask: list
            List of `Mask` objects that represent the mask in each band.
        variance: list
            List of `Image` objects that represent the variance in each band.

        Returns
        -------
        singles: list
            List of `MaskedImage` objects for each band,
            where the `image`, `mask`, and `variance` of each `single`
            point to the multiband objects.
        """
        singles = []
        if image is None:
            image = self.image
        if mask is None:
            mask = self.mask
        if variance is None:
            variance = self.variance

        for n in range(len(image)):
            dtype = self.singleType.__name__[-1]
            imageType = getattr(afwMaskedImage, "MaskedImage"+dtype)
            maskedImage = imageType(image=image[n], mask=mask[n], variance=variance[n])
            single = self.singleType(maskedImage)
            singles.append(single)
        return tuple(singles)

    @classmethod
    def fromButler(cls, butler, filters, filterKwargs, *args, **kwargs):
        """Load a multiband exposure from a butler

        Because each band is stored in a separate exposure file,
        this method can be used to load all of the exposures for
        a given set of bands

        Parameters
        ----------
        butler: `Butler`
            Butler connection to use to load the single band
            calibrated images
        filters: list or str
            List of filter names for each band
        filterKwargs: dict
            Keyword arguments to initialize a new instance of an
            inherited class that are different for each filter.
            The keys are the names of the arguments and the values
            should also be dictionaries, with filter names as keys
            and the value of the argument for the given filter as values.
        args: list
            Arguments to the Butler.
        kwargs: dict
            Keyword arguments to pass to initialize a new instance of an
            inherited class that are the same in all bands.

        Returns
        -------
        result: `MultibandExposure`
            The new `MultibandExposure` created by combining all of the
            single band exposures.
        """
        # Load the Exposure in each band
        exposures = []
        for f in filters:
            if filterKwargs is not None:
                for key, value in filterKwargs:
                    kwargs[key] = value[f]
            exposures.append(butler.get(*args, filter=f, **kwargs))
        return cls(filters, exposures)

    def setPsf(self, psf, filter):
        """Set the PSF for a single Exposure

        Parameters
        ----------
        psf: `meas.algorithms.coaddPsf`
            The PSF to assign to the given exposure
        filter: string or int
            Either the index of the filter or name of the
            filter for the `Exposure` in `self.singles`
            to assign `psf`.
        """
        if isinstance(filter, str):
            filter = self.filters.index(filter)
        self.singles[filter].setPsf(psf)
        # Clear the stored PSF image to be recalculated on demand later
        self._psfImage = None

    def setAllPsfs(self, psfs):
        """Set the PSF for each band

        Parameters
        ----------
        psfs: list of `meas.algorithms.coaddPsf`
            List of PSF's for each band
        """
        for psf in psfs:
            for single in self.singles:
                single.setPsf(psf)
        # Clear the stored PSF image to be recalculated on demand later
        self._psfImage = None

    def getPsfImage(self, recalculate=False):
        """Get a multiband PSF image

        If it has not been calculated already,
        the PSF Kernel Image is computed for each band
        and combined into a (filter, y, x) array and stored
        as `self._psfImage`.

        Parameters
        ----------
        recalculate: bool
            If the PSF kernel has already been calculated,
            if `recalculate` is `True` the psf image will
            be recalculated.

        Returns
        -------
        self._psfImage: array
            The multiband PSF image.
        """
        if recalculate or self._psfImage is None:
            psfs = []
            for single in self.singles:
                psfs.append(single.getPsf().computeKernelImage().array)
            self._psfImage = np.array(psfs)
        return self._psfImage
