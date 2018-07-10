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

__all__ = ["MultibandFootprint"]

import numpy as np

from lsst.geom import Point2I
from lsst.afw.geom import SpanSet
from lsst.afw.image import Mask, Image, MaskedImage, MultibandImage, MultibandMaskedImage
from lsst.afw.multiband import MultibandBase
from . import Footprint, makeHeavyFootprint


def getSpanSetFromImages(images, thresh=0, xy0=None):
    """Create a Footprint from a set of Images

    Parameters
    ----------
    images: `MultibandImage` or list of `Image`, array
        Images to extract the footprint from
    thresh: float
        All pixels above `thresh` will be included in the footprint
    xy0: `Point2I`
        Location of the minimum value of the images bounding box
        (if images is an array, otherwise the image bounding box is used).

    Returns
    -------
    spans: SpanSet
        Union of all spans in the images above the threshold
    imageBBox: Box2I
        Bounding box for the input images.
    """
    # Set the threshold for each band
    if not hasattr(thresh, "__len__"):
        thresh = [thresh] * len(images)

    # If images is a list of `afw Image` objects then
    # merge the SpanSet in each band into a single Footprint
    if isinstance(images, MultibandBase) or isinstance(images[0], Image):
        spans = SpanSet()
        for n, image in enumerate(images):
            mask = image.array > thresh[n]
            mask = Mask(mask.astype(np.int32), xy0=image.getBBox().getMin())
            spans = spans.union(SpanSet.fromMask(mask))
        imageBBox = images[0].getBBox()
    else:
        # Use thresh to detect the pixels above the threshold in each band
        thresh = np.array(thresh)
        if xy0 is None:
            xy0 = Point2I(0, 0)
        mask = np.any(images > thresh[:, None, None], axis=0)
        mask = Mask(mask.astype(np.int32), xy0=xy0)
        spans = SpanSet.fromMask(mask)
        imageBBox = mask.getBBox()
    return spans, imageBBox


def heavyFootprintToImage(heavy, fill=np.nan, bbox=None, imageType=MaskedImage):
    """Create an image of a HeavyFootprint

    Parameters
    ----------
    heavy: `HeavyFootprint`
        The HeavyFootprint to insert into the image
    fill: number
        Number to fill the pixels in the image that are not
        contained in `heavy`.
    bbox: `Box2I`
        Bounding box of the output image.
    imageType: type
        This should be either a `MaskedImage` or `Image` and describes
        the type of the output image.

    Returns
    -------
    image: `MaskedImage` or `Image`
        An image defined by `bbox` and padded with `fill` that
        contains the projected flux in `heavy`.
    """
    if bbox is None:
        bbox = heavy.getBBox()
    image = imageType(bbox, dtype=heavy.getImageArray().dtype)
    image.set(fill)
    heavy.insert(image)
    return image


class MultibandFootprint(MultibandBase):
    """Multiband Footprint class

    A `MultibandFootprint` is a collection of HeavyFootprints that have
    the same `SpanSet` and `peakCatalog` but different flux in each band.

    Parameters
    ----------
    filters: list
        List of filter names.
    footprint: `Footprint`
        `Footprint` that contains the `SpanSet` and `PeakCatalog`
        to use for the `HeavyFootprint` in each band.
    mMaskedImage: `MultibandMaskedImage`
        MultibandMaskedImage that footprint is a view into.
    """
    def __init__(self, filters, footprint, mMaskedImage):
        singles = [makeHeavyFootprint(footprint, mimg) for mimg in mMaskedImage]
        super().__init__(filters, singles)
        self._footprint = footprint
        self._mMaskedImage = mMaskedImage

    @staticmethod
    def fromArrays(filters, image, mask=None, variance=None, footprint=None, xy0=None, thresh=0, peaks=None):
        """Create a `MultibandFootprint` from an `image`, `mask`, `variance`

        Parameters
        ----------
        filters: list
            List of filter names.
        image: array
            An array to convert into `HeavyFootprint` objects.
            Only pixels above the `thresh` value for at least one band
            will be included in the `SpanSet` and resulting footprints.
        mask: array
            Mask for the `image` array.
        variance: array
            Variance of the `image` array.
        footprint: `Footprint`
            `Footprint` that contains the `SpanSet` and `PeakCatalog`
            to use for the `HeavyFootprint` in each band.
            If `footprint` is `None` then the `thresh` is used to create a
            `Footprint` based on the pixels above the `thresh` value.
        xy0: `Point2I`
            If `image` is an array and `footprint` is `None` then specifying
            `xy0` gives the location of the minimum `x` and `y` value of the
            `images`.
        thresh: float or list of floats
            Threshold in each band (or the same threshold to be used in all bands)
            to include a pixel in the `SpanSet` of the `MultibandFootprint`.
            If `Footprint` is not `None` then `thresh` is ignored.
        peaks: `PeakCatalog`
            Catalog containing information about the peaks located in the
            footprints.

        Returns
        -------
        result: MultibandFootprint
            MultibandFootprint created from the arrays
        """
        # Generate a new Footprint if one has not been specified
        if footprint is None:
            spans, imageBBox = getSpanSetFromImages(image, thresh, xy0)
            footprint = Footprint(spans)
        else:
            imageBBox = footprint.getBBox()

        if peaks is not None:
            footprint.setPeakCatalog(peaks)
        mMaskedImage = MultibandMaskedImage.fromArrays(filters, image, mask, variance, imageBBox)
        return MultibandFootprint(filters, footprint, mMaskedImage)

    @staticmethod
    def fromImages(filters, image, mask=None, variance=None, footprint=None, thresh=0, peaks=None):
        """Create a `MultibandFootprint` from an `image`, `mask`, `variance`

        Parameters
        ----------
        filters: list
            List of filter names.
        image: `MultibandImage`, or list of `Image`
            A `MultibandImage` (or collection of images in each band)
            to convert into `HeavyFootprint` objects.
            Only pixels above the `thresh` value for at least one band
            will be included in the `SpanSet` and resulting footprints.
        mask: `MultibandMask` or list of `Mask`
            Mask for the `image`.
        variance: `MultibandImage`, or list of `Image`
            Variance of the `image`.
        thresh: float or list of floats
            Threshold in each band (or the same threshold to be used in all bands)
            to include a pixel in the `SpanSet` of the `MultibandFootprint`.
            If `Footprint` is not `None` then `thresh` is ignored.
        peaks: `PeakCatalog`
            Catalog containing information about the peaks located in the
            footprints.

        Returns
        -------
        result: MultibandFootprint
            MultibandFootprint created from the image, mask, and variance
        """
        # Generate a new Footprint if one has not been specified
        if footprint is None:
            spans, imageBBox = getSpanSetFromImages(image, thresh)
            footprint = Footprint(spans)

        if peaks is not None:
            footprint.setPeakCatalog(peaks)
        mMaskedImage = MultibandMaskedImage(filters, image, mask, variance)
        return MultibandFootprint(filters, footprint, mMaskedImage)

    @staticmethod
    def fromHeavyFootprints(filters, heavies, fill=np.nan):
        """Build a `MultibandFootprint` from a list of `HeavyFootprint`s

        Each `HeavyFootprint` is expected to have the same `SpanSet`.

        Parameters
        ----------
        filters: list
            List of filter names.
        heavies: list
            A list of single band `HeavyFootprint` objects.
            Either `singles` or `images` must be specified.
            If `singles` is not `None`, then all arguments other than
            `filters` are ignored.
            Each `HeavyFootprint` should have the same `PeakCatalog` but
            is allowed to have a different `SpanSet`, in which case the
            `SpanSet` for each band is combined into the `SpanSet` of the
            `MultibandFootprint`.
        fill: fill: number
            Number to fill the pixels in the image that are not
            contained in a `HeavyFootprint`.

        Returns
        -------
        result: MultibandFootprint
            MultibandFootprint created from the heavy footprints
        """
        # Ensure that all HeavyFootprints have the same SpanSet
        spans = heavies[0].getSpans()
        if not all([heavy.getSpans() == spans for heavy in heavies]):
            raise ValueError("All HeavyFootprints in heavies are expected to have the same SpanSet")

        # Assume that all footprints have the same SpanSet and PeakCatalog
        footprint = Footprint(heavies[0].getSpans())
        footprint.setPeakCatalog(heavies[0].getPeaks())

        # Build the full masked images
        maskedImages = [heavyFootprintToImage(heavy, fill=fill) for heavy in heavies]
        mMaskedImage = MultibandMaskedImage.fromImages(filters, maskedImages)
        return MultibandFootprint(filters, footprint, mMaskedImage)

    def getSpans(self):
        """Get the full `SpanSet`"""
        return self._footprint.getSpans()

    @property
    def footprint(self):
        """Common SpanSet and peak catalog for the single band footprints"""
        return self._footprint

    @property
    def mMaskedImage(self):
        """MultibandMaskedImage that the footprints present a view into"""
        return self._mMaskedImage

    @property
    def spans(self):
        """`SpanSet` of the `MultibandFootprint`"""
        return self._footprint.getSpans()

    def getPeaks(self):
        """Get the `PeakCatalog`"""
        return self._footprint.getPeaks()

    @property
    def peaks(self):
        """`PeakCatalog` of the `MultibandFootprint`"""
        return self._footprint.getPeaks()

    def _slice(self, filters, filterIndex, indices):
        """Slice the current object and return the result

        `MultibandFootprint` objects cannot be sliced along the image
        dimension, so an error is thrown if `indices` has any elements.

        See `Multiband._slice` for a list of the parameters.
        """
        if len(indices) > 0:
            raise IndexError("MultibandFootprints can only be sliced in the filter dimension")

        if isinstance(filterIndex, slice):
            if filterIndex.start is not None:
                start = self.filters[filterIndex.start]
            else:
                start = None
            if filterIndex.stop is not None:
                stop = self.filters[filterIndex.stop]
            else:
                stop = None
            index = slice(start, stop, filterIndex.step)
        else:
            index = [self.filters[idx] for idx in filterIndex]
        mMaskedImage = self.mMaskedImage[index]
        assert mMaskedImage.filters == filters
        result = MultibandFootprint(filters, self.footprint, mMaskedImage)
        return result

    def getImage(self, bbox=None, fill=np.nan, imageType=MultibandMaskedImage):
        """Convert a `MultibandFootprint` to a `MultibandImage`

        This returns the heavy footprints converted into an `MultibandImage` or
        `MultibandMaskedImage` (depending on `imageType`).
        This might be different than the internal `mMaskedImage` property
        of the `MultibandFootprint`, as the `mMaskedImage` might contain
        some non-zero pixels not contained in the footprint but present in
        the images.

        Parameters
        ----------
        bbox: Box2I
            Bounding box of the resulting image.
            If no bounding box is specified, then the bounding box
            of the footprint is used.
        fill: float
            Value to use for any pixel in the resulting image
            outside of the `SpanSet`.
        imageType: type
            This should be either a `MultibandMaskedImage`
            or `MultibandImage` and describes the type of the output image.

        Returns
        -------
        result: MultibandBase
            The resulting `MultibandImage` or `MultibandMaskedImage` created
            from the `MultibandHeavyFootprint`.
        """
        if imageType == MultibandMaskedImage:
            singleType = MaskedImage
        elif imageType == MultibandImage:
            singleType = Image
        else:
            raise TypeError("Expected imageType to be either MultibandImage or MultibandMaskedImage")
        maskedImages = [heavyFootprintToImage(heavy, fill, bbox, singleType) for heavy in self.singles]
        mMaskedImage = imageType.fromImages(self.filters, maskedImages)
        return mMaskedImage

    def clone(self, deep=True):
        """Copy the current object

        Parameters
        ----------
        deep: bool
            Whether or not to make a deep copy

        Returns
        -------
        result: `MultibandFootprint`
            The cloned footprint.
        """
        if deep:
            footprint = Footprint(self.footprint.getSpans())
            for peak in self.footprint.getPeaks():
                footprint.addPeak(peak.getX(), peak.getY(), peak.getValue())
        else:
            footprint = self.footprint
        mMaskedImage = self.mMaskedImage.clone(deep)
        result = MultibandFootprint(self.filters, footprint, mMaskedImage)
        return result
