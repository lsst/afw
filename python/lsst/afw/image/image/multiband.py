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

__all__ = ["MultibandPixel", "MultibandImage", "MultibandMask", "MultibandMaskedImage"]

import numpy as np

from lsst.geom import Point2I, Box2I, Extent2I
from . import Image, ImageF, Mask, MaskPixel, PARENT, LOCAL
from ..maskedImage import MaskedImage, MaskedImageF
from ..slicing import imageIndicesToNumpy
from ...multiband import MultibandBase


class MultibandPixel(MultibandBase):
    """Multiband Pixel class

    This represent acts as a container for a single pixel
    (scalar) in multiple bands.

    Parameters
    ----------
    singles : `sequence`
       Either a list of single band objects or an array of values.
    filters : `list`
       List of filter names. If `singles` is an `OrderedDict` or
       a `MultibandPixel` then this argument is ignored,
       otherwise it is required.
    position : `Point2I`
       Location of the pixel in the parent image.
       Unlike other objects that inherit from `MultibandBase`,
       `MultibandPixel` objects don't have a full `Box2I`
       bounding box, since they only contain a single pixel,
       so the bounding box cannot be inherited from the
       list of `singles`.
    """
    def __init__(self, filters, singles, position):
        if any([arg is None for arg in [singles, filters, position]]):
            err = "Expected an array of `singles`, a list of `filters, and a `bbox`"
            raise NotImplementedError(err)

        # Make sure that singles is an array
        singles = np.array(singles, copy=False)

        super().__init__(filters, singles, bbox=Box2I(position, Extent2I(1, 1)))
        # In this case we want self.singles to be an array
        self._singles = singles

        # Make sure that the bounding box has been setup properly
        assert self.getBBox().getDimensions() == Extent2I(1, 1)

    def _getArray(self):
        """Data cube array in multiple bands

        Since `self._singles` is just a 1D array,
        `array` just returns `self._singles`.
        """
        return self.singles

    def _setArray(self, value):
        assert value.shape == self.array.shape
        self._singles[:] = value

    array = property(_getArray, _setArray)

    def clone(self, deep=True):
        """Make a copy of the current instance

        `MultibandPixel.singles` is an array,
        so this just makes a copy of the array
        (as opposed to a view of the parent array).
        """
        if deep:
            singles = np.copy(self.singles)
            position = self.getBBox().getMin()
            return MultibandPixel(filters=self.filters, singles=singles, position=position)

        result = MultibandPixel(filters=self.filters, singles=self.singles, position=self.getBBox().getMin())
        result._bbox = self.getBBox()
        return result

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
        filters, filterIndex = self._filterNamesToIndex(indices)
        if len(filters) == 1 and not isinstance(filterIndex, slice):
            # The user only requested a pixel in a single band, so return it
            return self.array[filterIndex[0]]

        result = self.clone(False)
        result._filters = filters
        result._singles = self._singles[filterIndex]
        # No need to update the bounding box, since pixels can only be sliced in the filter dimension
        return result

    def _slice(self, filters, filterIndex, indices):
        pass


class MultibandImageBase(MultibandBase):
    """Multiband Image class

    This class acts as a container for multiple `afw.Image` objects.
    All images must be contained in the same bounding box,
    and have the same data type.
    The data is stored in a 3D array (filters, y, x), and the single
    band `Image` instances have an internal array that points to the
    3D multiband array, so that the single band objects and multiband
    array are always in agreement.

    Parameters
    ----------
    filters : `list`
       List of filter names.
    array : 3D numpy array
       Array (filters, y, x) of multiband data.
       If this is used to initialize a `MultibandImage`,
       either `bbox` or `singles` is also required.
    singleType : `type`
       Type of the single band object (eg. `Image`, `Mask`) to
       convert the array into a tuple of single band objects
       that point to the image array.
    bbox : `Box2I`
       Location of the array in a larger single band image.
       If `bbox` is `None` then the bounding box is initialized
       at the origin.
    """
    def __init__(self, filters, array, singleType, bbox=None):
        # Create the single band objects from the array
        if len(array) != len(filters):
            raise ValueError("`array` and `filters` must have the same length")
        self._array = array
        self._filters = tuple(filters)
        if bbox is None:
            bbox = Box2I(Point2I(0, 0), Extent2I(array.shape[2], array.shape[1]))
        self._bbox = bbox

        xy0 = self.getXY0()
        dtype = array.dtype
        self._singles = tuple([singleType(array=array[n], xy0=xy0, dtype=dtype) for n in range(len(array))])

        # Make sure that all of the parameters have been setup appropriately
        assert isinstance(self._bbox, Box2I)
        assert len(self.singles) == len(self.filters)

    def _getArray(self):
        """Data cube array in multiple bands

        Returns
        -------
        self._array : array
           The resulting 3D data cube with shape (filters, y, x).
        """
        return self._array

    def _setArray(self, value):
        """Set the values of the array"""
        self.array[:] = value

    array = property(_getArray, _setArray)

    def clone(self, deep=True):
        """Copy the current object

        Parameters
        ----------
        deep : `bool`
           Whether or not to make a deep copy
        """
        if deep:
            array = np.copy(self.array)
            bbox = Box2I(self.getBBox())
        else:
            array = self.array
            bbox = self.getBBox()
        result = type(self)(self.filters, array, bbox)
        return result

    def _slice(self, filters, filterIndex, indices):
        """Slice the current object and return the result

        See `MultibandBase._slice` for a list of the parameters.
        """
        if len(indices) > 0:
            if len(indices) == 1:
                indices = indices[0]
            allSlices = [filterIndex, slice(None), slice(None)]
            sy, sx, bbox = imageIndicesToNumpy(indices, self.getBBox)
            if sy is not None:
                allSlices[-2] = sy
            if sx is not None:
                allSlices[-1] = sx
            array = self._array[tuple(allSlices)]

            # Return a scalar or MultibandPixel
            # if the image indices are integers
            if bbox is None:
                if not isinstance(filterIndex, slice) and len(filterIndex) == 1:
                    return array[0]
                result = MultibandPixel(
                    singles=array,
                    filters=filters,
                    position=Point2I(sx + self.x0, sy + self.y0)
                )
                return result
        else:
            array = self._array[filterIndex]
            bbox = self.getBBox()
        result = type(self)(filters, array, bbox)

        # Check that the image and array shapes agree
        imageShape = (
            len(result.filters),
            result.getBBox().getHeight(),
            result.getBBox().getWidth()
        )
        assert result.array.shape == imageShape

        return result

    def __setitem__(self, args, value):
        """Set a subset of the MultibandImage
        """
        if not isinstance(args, tuple):
            indices = (args,)
        else:
            indices = args

        # Return the single band object if the first
        # index is not a list or slice.
        filters, filterIndex = self._filterNamesToIndex(indices[0])
        if len(indices) > 1:
            sy, sx, bbox = imageIndicesToNumpy(indices[1:], self.getBBox)
        else:
            sy = sx = slice(None)
        if hasattr(value, "array"):
            self._array[filterIndex, sy, sx] = value.array
        else:
            self._array[filterIndex, sy, sx] = value

    def getBBox(self, origin=PARENT):
        """Bounding box
        """
        if origin == PARENT:
            return self._bbox
        elif origin == LOCAL:
            return Box2I(Point2I(0, 0), self._bbox.getDimensions())
        raise ValueError("Unrecognized origin, expected either PARENT or LOCAL")


def makeImageFromSingles(cls, filters, singles):
    """Construct a MultibandImage from a collection of single band images

    Parameters
    ----------
    filters : `list`
       List of filter names.
    singles : `list`
       A list of single band objects.
       If `array` is not `None`, then `singles` is ignored
    """
    array = np.array([image.array for image in singles], dtype=singles[0].array.dtype)
    if not np.all([image.getBBox() == singles[0].getBBox() for image in singles[1:]]):
        raise ValueError("Single band images did not all have the same bounding box")
    bbox = singles[0].getBBox()
    return cls(filters, array, bbox)


def makeImageFromKwargs(cls, filters, filterKwargs, singleType=ImageF, **kwargs):
    """Build a MultibandImage from a set of keyword arguments

    Parameters
    ----------
    filters : `list`
       List of filter names.
    singleType : class
       Class of the single band objects.
       This is ignored unless `singles` and `array`
       are both `None`, in which case it is required.
    filterKwargs : `dict`
       Keyword arguments to initialize a new instance of an inherited class
       that are different for each filter.
       The keys are the names of the arguments and the values
       should also be dictionaries, with filter names as keys
       and the value of the argument for a given filter as values.
    kwargs : `dict`
       Keyword arguments to initialize a new instance of an
       inherited class that are the same in all bands.
    """
    # Attempt to load a set of images
    singles = []
    for f in filters:
        if filterKwargs is not None:
            for key, value in filterKwargs:
                kwargs[key] = value[f]
        singles.append(singleType(**kwargs))
    return cls.makeImageFromSingles(filters, singles)


class MultibandImage(MultibandImageBase):
    """Multiband Image class

    See `MultibandImageBase` for a description of the parameters.
    """
    def __init__(self, filters, array, bbox=None):
        super().__init__(filters, array, Image, bbox)

    @staticmethod
    def fromImages(filters, singles):
        """Construct a MultibandImage from a collection of single band images

        see `fromSingles` for a description of parameters
        """
        return makeImageFromSingles(MultibandImage, filters, singles)

    @staticmethod
    def fromKwargs(filters, filterKwargs, singleType=ImageF, **kwargs):
        """Build a MultibandImage from a set of keyword arguments

        see `makeImageFromKwargs` for a description of parameters
        """
        return makeImageFromKwargs(MultibandImage, filters, filterKwargs, singleType, **kwargs)


class MultibandMask(MultibandImageBase):
    """Multiband Mask class

    See `MultibandImageBase` for a description of the parameters.
    """
    def __init__(self, filters, array, bbox=None):
        super().__init__(filters, array, Mask, bbox)
        # Set Mask specific properties
        self._refMask = self._singles[0]
        refMask = self._refMask
        assert np.all([refMask.getMaskPlaneDict() == m.getMaskPlaneDict() for m in self.singles])
        assert np.all([refMask.getNumPlanesMax() == m.getNumPlanesMax() for m in self.singles])
        assert np.all([refMask.getNumPlanesUsed() == m.getNumPlanesUsed() for m in self.singles])

    @staticmethod
    def fromMasks(filters, singles):
        """Construct a MultibandImage from a collection of single band images

        see `fromSingles` for a description of parameters
        """
        return makeImageFromSingles(MultibandMask, filters, singles)

    @staticmethod
    def fromKwargs(filters, filterKwargs, singleType=ImageF, **kwargs):
        """Build a MultibandImage from a set of keyword arguments

        see `makeImageFromKwargs` for a description of parameters
        """
        return makeImageFromKwargs(MultibandMask, filters, filterKwargs, singleType, **kwargs)

    def getMaskPlane(self, key):
        """Get the bit number of a mask in the `MaskPlaneDict`

        Each `key` in the mask plane has an associated bit value
        in the mask. This method returns the bit number of the
        `key` in the `MaskPlaneDict`.
        This is in contrast to `getPlaneBitMask`, which returns the
        value of the bit number.

        For example, if `getMaskPlane` returns `8`, then `getPlaneBitMask`
        returns `256`.

        Parameters
        ----------
        key : `str`
           Name of the key in the `MaskPlaneDict`

        Returns
        -------
        bit : `int`
           Bit number for mask `key`
        """
        return self._refMask.getMaskPlaneDict()[key]

    def getPlaneBitMask(self, names):
        """Get the bit number of a mask in the `MaskPlaneDict`

        Each `key` in the mask plane has an associated bit value
        in the mask. This method returns the bit number of the
        `key` in the `MaskPlaneDict`.
        This is in contrast to `getPlaneBitMask`, which returns the
        value of the bit number.

        For example, if `getMaskPlane` returns `8`, then `getPlaneBitMask`
        returns `256`.

        Parameters
        ----------
        names : `str` or list of `str`
           Name of the key in the `MaskPlaneDict` or a list of keys.
           If multiple keys are used, the value returned is the integer
           value of the number with all of the bit values in `names`.

           For example if `MaskPlaneDict("CR")=3` and
           `MaskPlaneDict("NO_DATA)=8`, then
           `getPlaneBitMask(("CR", "NO_DATA"))=264`

        Returns
        -------
        bit value : `int`
           Bit value for all of the combined bits described by `names`.
        """
        return self._refMask.getPlaneBitMask(names)

    def getNumPlanesMax(self):
        """Maximum number of mask planes available

        This is required to be the same for all of the single
        band `Mask` objects.
        """
        return self._refMask.getNumPlanesMax()

    def getNumPlanesUsed(self):
        """Number of mask planes used

        This is required to be the same for all of the single
        band `Mask` objects.
        """
        return self._refMask.getNumPlanesUsed()

    def getMaskPlaneDict(self):
        """Dictionary of Mask Plane bit values
        """
        return self._refMask.getMaskPlaneDict()

    @staticmethod
    def clearMaskPlaneDict():
        """Reset the mask plane dictionary
        """
        Mask[MaskPixel].clearMaskPlaneDict()

    @staticmethod
    def addMaskPlane(name):
        """Add a mask to the mask plane

        Parameters
        ----------
        name : `str`
           Name of the new mask plane

        Returns
        -------
        index : `int`
           Bit value of the mask in the mask plane.
        """
        idx = Mask[MaskPixel].addMaskPlane(name)
        return idx

    @staticmethod
    def removeMaskPlane(name):
        """Remove a mask from the mask plane

        Parameters
        ----------
        name : `str`
           Name of the mask plane to remove
        """
        Mask[MaskPixel].removeMaskPlane(name)

    def removeAndClearMaskPlane(self, name, removeFromDefault=False):
        """Remove and clear a mask from the mask plane

        Clear all pixels of the specified mask and remove the plane from the
        mask plane dictionary.  Also optionally remove the plane from the
        default dictionary.

        Parameters
        ----------
        name : `str`
           Name of the mask plane to remove
        removeFromDefault : `bool`, optional
           Whether to remove the mask plane from the default dictionary.
           Default is `False`.
        """
        # Clear all masks in MultibandMask but leave in default dict for now
        for single in self.singles:
            single.removeAndClearMaskPlane(name, removeFromDefault=False)
        # Now remove from default dict according to removeFromDefault
        self._refMask.removeAndClearMaskPlane(name, removeFromDefault)

    def clearAllMaskPlanes(self):
        """Clear all the pixels
        """
        self._refMask.clearAllMaskPlanes()

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
        for s, o in zip(self.singles, _others):
            s |= o
        return self

    def __iand__(self, others):
        _others = self._getOtherMasks(others)
        for s, o in zip(self.singles, _others):
            s &= o
        return self

    def __ixor__(self, others):
        _others = self._getOtherMasks(others)
        for s, o in zip(self.singles, _others):
            s ^= o
        return self


class MultibandTripleBase(MultibandBase):
    """MultibandTripleBase class

    This is a base class inherited by multiband classes
    with `image`, `mask`, and `variance` objects,
    such as `MultibandMaskedImage` and `MultibandExposure`.

    Parameters
    ----------
    filters : `list`
       List of filter names. If `singles` is an `OrderedDict`
       then this argument is ignored, otherwise it is required.
    image : `list` or `MultibandImage`
       List of `Image` objects that represent the image in each band or
       a `MultibandImage`.
       Ignored if `singles` is not `None`.
    mask : `list` or `MultibandMask`
       List of `Mask` objects that represent the mask in each bandor
       a `MultibandMask`.
       Ignored if `singles` is not `None`.
    variance : `list` or `MultibandImage`
       List of `Image` objects that represent the variance in each bandor
       a `MultibandImage`.
       Ignored if `singles` is not `None`.
    """
    def __init__(self, filters, image, mask, variance):
        self._filters = tuple(filters)
        # Convert single band images into multiband images
        if not isinstance(image, MultibandBase):
            image = MultibandImage.fromImages(filters, image)
            if mask is not None:
                mask = MultibandMask.fromMasks(filters, mask)
            if variance is not None:
                variance = MultibandImage.fromImages(filters, variance)
        self._image = image
        self._mask = mask
        self._variance = variance

        self._singles = self._buildSingles(self._image, self._mask, self._variance)
        self._bbox = self.singles[0].getBBox()

    def setXY0(self, xy0):
        """Shift the bounding box but keep the same Extent
        This is different than `MultibandBase.setXY0`
        because the multiband `image`, `mask`, and `variance` objects
        must all have their bounding boxes updated.
        Parameters
        ----------
        xy0 : `Point2I`
           New minimum bounds of the bounding box
        """
        super().setXY0(xy0)
        self.image.setXY0(xy0)
        if self.mask is not None:
            self.mask.setXY0(xy0)
        if self.variance is not None:
            self.variance.setXY0(xy0)

    def shiftedTo(self, xy0):
        """Shift the bounding box but keep the same Extent

        This is different than `MultibandBase.shiftedTo`
        because the multiband `image`, `mask`, and `variance` objects
        must all have their bounding boxes updated.

        Parameters
        ----------
        xy0 : `Point2I`
           New minimum bounds of the bounding box

        Returns
        -------
        result : `MultibandBase`
           A copy of the object, shifted to `xy0`.
        """
        raise NotImplementedError("shiftedTo not implemented until DM-10781")
        result = self.clone(False)
        result._image = result.image.shiftedTo(xy0)
        if self.mask is not None:
            result._mask = result.mask.shiftedTo(xy0)
        if self.variance is not None:
            result._variance = result.variance.shiftedTo(xy0)
        result._bbox = result.image.getBBox()
        return result

    def clone(self, deep=True):
        """Make a copy of the current instance
        """
        image = self.image.clone(deep)
        if self.mask is not None:
            mask = self.mask.clone(deep)
        else:
            mask = None
        if self.variance is not None:
            variance = self.variance.clone(deep)
        else:
            variance = None
        return type(self)(self.filters, image, mask, variance)

    def _slice(self, filters, filterIndex, indices):
        """Slice the current object and return the result

        See `Multiband._slice` for a list of the parameters.
        """
        image = self.image._slice(filters, filterIndex, indices)
        if self.mask is not None:
            mask = self._mask._slice(filters, filterIndex, indices)
        else:
            mask = None
        if self.variance is not None:
            variance = self._variance._slice(filters, filterIndex, indices)
        else:
            variance = None

        # If only a single pixel is selected, return the tuple of MultibandPixels
        if isinstance(image, MultibandPixel):
            if mask is not None:
                assert isinstance(mask, MultibandPixel)
            if variance is not None:
                assert isinstance(variance, MultibandPixel)
            return (image, mask, variance)

        result = type(self)(filters=filters, image=image, mask=mask, variance=variance)
        assert all([r.getBBox() == result._bbox for r in [result._mask, result._variance]])
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

    @property
    def image(self):
        """The image of the MultibandMaskedImage"""
        return self._image

    @property
    def mask(self):
        """The mask of the MultibandMaskedImage"""
        return self._mask

    @property
    def variance(self):
        """The variance of the MultibandMaskedImage"""
        return self._variance

    def getBBox(self, origin=PARENT):
        """Bounding box
        """
        if origin == PARENT:
            return self._bbox
        elif origin == LOCAL:
            return Box2I(Point2I(0, 0), self._bbox.getDimensions())
        raise ValueError("Unrecognized origin, expected either PARENT or LOCAL")


def tripleFromSingles(cls, filters, singles, **kwargs):
    """Construct a MultibandTriple from a collection of single band objects

    Parameters
    ----------
    filters : `list`
       List of filter names.
    singles : `list`
       A list of single band objects.
       If `array` is not `None`, then `singles` is ignored
    """
    if not np.all([single.getBBox() == singles[0].getBBox() for single in singles[1:]]):
        raise ValueError("Single band images did not all have the same bounding box")
    image = MultibandImage.fromImages(filters, [s.image for s in singles])
    mask = MultibandMask.fromMasks(filters, [s.mask for s in singles])
    variance = MultibandImage.fromImages(filters, [s.variance for s in singles])
    return cls(filters, image, mask, variance, **kwargs)


def tripleFromArrays(cls, filters, image, mask, variance, bbox=None):
    """Construct a MultibandTriple from a set of arrays

    Parameters
    ----------
    filters : `list`
       List of filter names.
    image : array
       Array of image values
    mask : array
       Array of mask values
    variance : array
       Array of variance values
    bbox : `Box2I`
       Location of the array in a larger single band image.
       This argument is ignored if `singles` is not `None`.
    """
    if bbox is None:
        bbox = Box2I(Point2I(0, 0), Extent2I(image.shape[1], image.shape[0]))
    mImage = MultibandImage(filters, image, bbox)
    if mask is not None:
        mMask = MultibandMask(filters, mask, bbox)
    else:
        mMask = None
    if variance is not None:
        mVariance = MultibandImage(filters, variance, bbox)
    else:
        mVariance = None
    return cls(filters, mImage, mMask, mVariance)


def makeTripleFromKwargs(cls, filters, filterKwargs, singleType, **kwargs):
    """Build a MultibandImage from a set of keyword arguments

    Parameters
    ----------
    filters : `list`
       List of filter names.
    singleType : `class`
       Class of the single band objects.
       This is ignored unless `singles` and `array`
       are both `None`, in which case it is required.
    filterKwargs : `dict`
       Keyword arguments to initialize a new instance of an inherited class
       that are different for each filter.
       The keys are the names of the arguments and the values
       should also be dictionaries, with filter names as keys
       and the value of the argument for a given filter as values.
    kwargs : `dict`
       Keyword arguments to initialize a new instance of an inherited
       class that are the same in all bands.
    """
    # Attempt to load a set of images
    singles = []
    for f in filters:
        if filterKwargs is not None:
            for key, value in filterKwargs:
                kwargs[key] = value[f]
        singles.append(singleType(**kwargs))
    return tripleFromSingles(cls, filters, singles)


class MultibandMaskedImage(MultibandTripleBase):
    """MultibandMaskedImage class

    This class acts as a container for multiple `afw.MaskedImage` objects.
    All masked images must have the same bounding box, and the associated
    images must all have the same data type.
    The `image`, `mask`, and `variance` are all stored separately into
    a `MultibandImage`, `MultibandMask`, and `MultibandImage` respectively,
    which each have their own internal 3D arrays (filter, y, x).

    See `MultibandTripleBase` for parameter definitions.
    """
    def __init__(self, filters, image=None, mask=None, variance=None):
        super().__init__(filters, image, mask, variance)

    @staticmethod
    def fromImages(filters, singles):
        """Construct a MultibandImage from a collection of single band images

        see `tripleFromImages` for a description of parameters
        """
        return tripleFromSingles(MultibandMaskedImage, filters, singles)

    @staticmethod
    def fromArrays(filters, image, mask, variance, bbox=None):
        """Construct a MultibandMaskedImage from a collection of arrays

        see `tripleFromArrays` for a description of parameters
        """
        return tripleFromArrays(MultibandMaskedImage, filters, image, mask, variance, bbox)

    @staticmethod
    def fromKwargs(filters, filterKwargs, singleType=MaskedImageF, **kwargs):
        """Build a MultibandImage from a set of keyword arguments

        see `makeTripleFromKwargs` for a description of parameters
        """
        return makeTripleFromKwargs(MultibandMaskedImage, filters, filterKwargs, singleType, **kwargs)

    def _buildSingles(self, image=None, mask=None, variance=None):
        """Make a new list of single band objects

        Parameters
        ----------
        image : `MultibandImage`
           `MultibandImage` object that represent the image in each band.
        mask : `MultibandMask`
           `MultibandMask` object that represent the mask in each band.
        variance : `MultibandImage`
           `MultibandImage` object that represent the variance in each band.

        Returns
        -------
        singles : `tuple`
           Tuple of `MaskedImage` objects for each band,
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

        dtype = image.array.dtype
        singles = []
        for f in self.filters:
            _image = image[f]
            if mask is not None:
                _mask = mask[f]
            else:
                _mask = None
            if variance is not None:
                _variance = variance[f]
            else:
                _variance = None
            singles.append(MaskedImage(image=_image, mask=_mask, variance=_variance, dtype=dtype))
        return tuple(singles)
