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

# Need to use the private name of the package for imports here because
# lsst.afw.image itself imports lsst.afw.detection
from lsst.afw.image._image import Mask, Image, MultibandImage, MultibandMaskedImage
from lsst.afw.image._maskedImage import MaskedImage

from lsst.afw.multiband import MultibandBase
from . import Footprint, makeHeavyFootprint


def getSpanSetFromImages(images, thresh=0, xy0=None):
    """Create a Footprint from a set of Images

    Parameters
    ----------
    images : `lsst.afw.image.MultibandImage` or list of `lsst.afw.image.Image`, array
        Images to extract the footprint from
    thresh : `float`
        All pixels above `thresh` will be included in the footprint
    xy0 : `lsst.geom.Point2I`
        Location of the minimum value of the images bounding box
        (if images is an array, otherwise the image bounding box is used).

    Returns
    -------
    spans : `lsst.afw.geom.SpanSet`
        Union of all spans in the images above the threshold
    imageBBox : `lsst.afw.detection.Box2I`
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


class MultibandFootprint(MultibandBase):
    """Multiband Footprint class

    A `MultibandFootprint` is a collection of HeavyFootprints that have
    the same `SpanSet` and `peakCatalog` but different flux in each band.

    Parameters
    ----------
    bands : `list`
        List of band names.
    singles : `list`
        A list of single band `HeavyFootprint` objects.
        Each `HeavyFootprint` should have the same `PeakCatalog`
        and the same `SpanSet`, however to save CPU cycles there
        is no internal check for consistency of the peak catalog.
    """
    def __init__(self, bands, singles):
        super().__init__(bands, singles)
        # Ensure that all HeavyFootprints have the same SpanSet
        spans = singles[0].getSpans()
        if not all([heavy.getSpans() == spans for heavy in singles]):
            raise ValueError("All HeavyFootprints in singles are expected to have the same SpanSet")

        # Assume that all footprints have the same SpanSet and PeakCatalog
        footprint = Footprint(spans)
        footprint.setPeakCatalog(singles[0].getPeaks())
        self._footprint = footprint

    @staticmethod
    def fromArrays(bands, image, mask=None, variance=None, footprint=None, xy0=None, thresh=0, peaks=None):
        """Create a `MultibandFootprint` from an `image`, `mask`, `variance`

        Parameters
        ----------
        bands : `list`
            List of band names.
        image: array
            An array to convert into `lsst.afw.detection.HeavyFootprint` objects.
            Only pixels above the `thresh` value for at least one band
            will be included in the `SpanSet` and resulting footprints.
        mask : array
            Mask for the `image` array.
        variance : array
            Variance of the `image` array.
        footprint : `Footprint`
            `Footprint` that contains the `SpanSet` and `PeakCatalog`
            to use for the `HeavyFootprint` in each band.
            If `footprint` is `None` then the `thresh` is used to create a
            `Footprint` based on the pixels above the `thresh` value.
        xy0 : `Point2I`
            If `image` is an array and `footprint` is `None` then specifying
            `xy0` gives the location of the minimum `x` and `y` value of the
            `images`.
        thresh : `float` or list of floats
            Threshold in each band (or the same threshold to be used in all bands)
            to include a pixel in the `SpanSet` of the `MultibandFootprint`.
            If `Footprint` is not `None` then `thresh` is ignored.
        peaks : `PeakCatalog`
            Catalog containing information about the peaks located in the
            footprints.

        Returns
        -------
        result : `MultibandFootprint`
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
        mMaskedImage = MultibandMaskedImage.fromArrays(bands, image, mask, variance, imageBBox)
        singles = [makeHeavyFootprint(footprint, maskedImage) for maskedImage in mMaskedImage]
        return MultibandFootprint(bands, singles)

    @staticmethod
    def fromImages(bands, image, mask=None, variance=None, footprint=None, thresh=0, peaks=None):
        """Create a `MultibandFootprint` from an `image`, `mask`, `variance`

        Parameters
        ----------
        bands : `list`
            List of band names.
        image : `lsst.afw.image.MultibandImage`, or list of `lsst.afw.image.Image`
            A `lsst.afw.image.MultibandImage` (or collection of images in each band)
            to convert into `HeavyFootprint` objects.
            Only pixels above the `thresh` value for at least one band
            will be included in the `SpanSet` and resulting footprints.
        mask : `MultibandMask` or list of `Mask`
            Mask for the `image`.
        variance : `lsst.afw.image.MultibandImage`, or list of `lsst.afw.image.Image`
            Variance of the `image`.
        thresh : `float` or `list` of floats
            Threshold in each band (or the same threshold to be used in all bands)
            to include a pixel in the `SpanSet` of the `MultibandFootprint`.
            If `Footprint` is not `None` then `thresh` is ignored.
        peaks : `PeakCatalog`
            Catalog containing information about the peaks located in the
            footprints.

        Returns
        -------
        result : `MultibandFootprint`
            MultibandFootprint created from the image, mask, and variance
        """
        # Generate a new Footprint if one has not been specified
        if footprint is None:
            spans, imageBBox = getSpanSetFromImages(image, thresh)
            footprint = Footprint(spans)

        if peaks is not None:
            footprint.setPeakCatalog(peaks)
        mMaskedImage = MultibandMaskedImage(bands, image, mask, variance)
        singles = [makeHeavyFootprint(footprint, maskedImage) for maskedImage in mMaskedImage]
        return MultibandFootprint(bands, singles)

    @staticmethod
    def fromMaskedImages(bands, maskedImages, footprint=None, thresh=0, peaks=None):
        """Create a `MultibandFootprint` from a list of `MaskedImage`

        See `fromImages` for a description of the parameters not listed below

        Parameters
        ----------
        maskedImages : `list` of `lsst.afw.image.MaskedImage`
            MaskedImages to extract the single band heavy footprints from.
            Like `fromImages`, if a `footprint` is not specified then all
            pixels above `thresh` will be used, and `peaks` will be added
            to the `PeakCatalog`.

        Returns
        -------
        result : `MultibandFootprint`
            MultibandFootprint created from the image, mask, and variance
        """
        image = [maskedImage.image for maskedImage in maskedImages]
        mask = [maskedImage.mask for maskedImage in maskedImages]
        variance = [maskedImage.variance for maskedImage in maskedImages]
        return MultibandFootprint.fromImages(bands, image, mask, variance, footprint, thresh, peaks)

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

    def _slice(self, bands, bandIndex, indices):
        """Slice the current object and return the result

        `MultibandFootprint` objects cannot be sliced along the image
        dimension, so an error is thrown if `indices` has any elements.

        See `Multiband._slice` for a list of the parameters.
        """
        if len(indices) > 0:
            raise IndexError("MultibandFootprints can only be sliced in the band dimension")

        if isinstance(bandIndex, slice):
            singles = self.singles[bandIndex]
        else:
            singles = [self.singles[idx] for idx in bandIndex]

        return MultibandFootprint(bands, singles)

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
        bbox : `Box2I`
            Bounding box of the resulting image.
            If no bounding box is specified, then the bounding box
            of the footprint is used.
        fill : `float`
            Value to use for any pixel in the resulting image
            outside of the `SpanSet`.
        imageType : `type`
            This should be either a `MultibandMaskedImage`
            or `MultibandImage` and describes the type of the output image.

        Returns
        -------
        result : `MultibandBase`
            The resulting `MultibandImage` or `MultibandMaskedImage` created
            from the `MultibandHeavyFootprint`.
        """
        if imageType == MultibandMaskedImage:
            singleType = MaskedImage
        elif imageType == MultibandImage:
            singleType = Image
        else:
            raise TypeError("Expected imageType to be either MultibandImage or MultibandMaskedImage")
        maskedImages = [heavy.extractImage(fill, bbox, singleType) for heavy in self.singles]
        mMaskedImage = imageType.fromImages(self.bands, maskedImages)
        return mMaskedImage

    def clone(self, deep=True):
        """Copy the current object

        Parameters
        ----------
        deep : `bool`
            Whether or not to make a deep copy

        Returns
        -------
        result : `MultibandFootprint`
            The cloned footprint.
        """
        if deep:
            footprint = Footprint(self.footprint.getSpans())
            for peak in self.footprint.getPeaks():
                footprint.addPeak(peak.getX(), peak.getY(), peak.getValue())
            mMaskedImage = self.getImage()
            bands = tuple([f for f in self.bands])
            result = MultibandFootprint.fromMaskedImages(bands, mMaskedImage, footprint)
        else:
            result = MultibandFootprint(self.bands, self.singles)
        return result
