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
from lsst.afw.image import Mask, Image, MaskedImage, MultibandImage
from lsst.afw.multiband import MultibandBase
from . import Footprint, makeHeavyFootprint


class MultibandFootprint(MultibandBase):
    """Multiband Footprint class

    A `MultibandFootprint` is a collection of HeavyFootprints that have
    the same `SpanSet` and `peakCatalog` but different flux in each band.

    Parameters
    ----------
    filters: list
        List of filter names.
        If `singles` is an `OrderedDict` then this argument is ignored,
        otherwise it is required.
    singles: list
        A list of single band `HeavyFootprint` objects.
        Either `singles` or `images` must be specified.
        If `singles` is not `None`, then all arguments other than
        `filters` are ignored.
        Each `HeavyFootprint` should have the same `PeakCatalog` but
        is allowed to have a different `SpanSet`, in which case the
        `SpanSet` for each band is combined into the `SpanSet` of the
        `MultibandFootprint`.
    images: array, `MultibandImage`, or list of `afw.Image` objects
        An array or `MultibandImage` (or collection of images in each band)
        to convert into `HeavyFootprint` objects.
        Only pixels above the `thresh` value for at least one band
        will be included in the `SpanSet` and resulting footprints.
    footprint: `Footprint`
        `Footprint` that contains the `SpanSet` and `PeakCatalog`
        to use for the `HeavyFootprint` in each band.
    xy0: `Point2I`
        If `images` is an array and `footprint` is `None` then specifying
        `xy0` gives the location of the minimum `x` and `y` value of the
        `images`.
    peaks: `PeakCatalog`
        Catalog containing information about the peaks located in the
        footprints.
    thresh: float or list of floats
        Threshold in each band (or the same threshold to be used in all bands)
        to include a pixel in the `SpanSet` of the `MultibandFootprint`.
        If `Footprint` is not `None` then `thresh` is ignored.
    """
    def __init__(self, filters, singles=None, images=None, footprint=None, xy0=None, peaks=None, thresh=0):
        if singles is None and images is not None:
            from lsst.afw.image.utils import projectImage
            if len(images) != len(filters):
                err = "`filters` and `images` should have the same length, got {0}, {1}"
                raise ValueError(err.format(len(filters), len(images)))
            # Create a Heavy Footprint in each band
            if footprint is None:
                # Set the threshold for each band
                if not hasattr(thresh, "__len__"):
                    thresh = [thresh] * len(filters)

                # If images is a list of `afw Image` objects then
                # merge the SpanSet in each band into a single Footprint
                if isinstance(images[0], Image):
                    spans = SpanSet()
                    for n, image in enumerate(images):
                        mask = image.array > thresh[n]
                        mask = Mask(mask.astype(np.int32), xy0=image.getBBox().getMin())
                        footprint = Footprint(SpanSet.fromMask(mask))
                        spans = spans.union(footprint.getSpans())
                    self._footprint = Footprint(spans)
                else:
                    # Use thresh to detect the pixels above the threshold in each band
                    thresh = np.array(thresh)
                    if xy0 is None:
                        xy0 = Point2I(0, 0)
                    mask = np.any(images > thresh[:, None, None], axis=0)
                    mask = Mask(mask.astype(np.int32), xy0=xy0)
                    self._footprint = Footprint(SpanSet.fromMask(mask))
                if peaks is not None:
                    self._footprint.setPeakCatalog(peaks)
            else:
                self._footprint = footprint
            if not isinstance(images[0], MaskedImage):
                if not isinstance(images[0], Image):
                    images = [Image(i, dtype=i.dtype, xy0=xy0) for i in images]
                images = [MaskedImage(i, dtype=i.dtype) for i in images]

            bbox = self._footprint.getBBox()
            self._bbox = bbox
            images = [projectImage(image, bbox) for image in images]
            for n, image in enumerate(images):
                if image.getBBox() != bbox:
                    newImage = type(image.image)(bbox)
                    newMask = type(image.mask)(bbox)
                    newVariance = type(image.variance)(bbox)
                    print(image.getBBox())
                    sy, sx = self.imageIndicesToNumpy(image.getBBox())
                    print(sy, sx, newImage.getBBox())
                    newImage.array[sy, sx] += image.image.array
                    newMask.array[sy, sx] += image.mask.array
                    newVariance.array[sy, sx] += image.variance.array
                    images[n] = MaskedImage(image=newImage, mask=newMask, variance=newVariance,
                                            dtype=newImage.array.dtype)
            singles = [makeHeavyFootprint(self._footprint, images[n]) for n in range(len(filters))]
        elif singles is not None:
            # Verify that all of the `HeavyFootprint`s have the same PeakCatalog
            peaks = singles[0].getPeaks()
            for single in singles[1:]:
                _peaks = single.getPeaks()
                if not np.all([peaks[key] == _peaks[key] for key in ["id", "f_x", "f_y"]]):
                    raise ValueError("All heavy footprints should have the same peak catalog")
            # Build a common Footprint for all bands
            spans = singles[0].getSpans()
            for single in singles[1:]:
                spans = spans.union(single.getSpans())
            self._footprint = Footprint(spans)
            self._footprint.setPeakCatalog(peaks)
        else:
            err = ("MultibandFootprint must be initialized with a list of filters and either "
                   "a list of single band HeavyFootprints (`singles`) or `images` ")
            raise ValueError(err)

        self._filters = filters
        self._singles = singles
        self._singleType = type(self._singles[0])
        self._bbox = self._footprint.getBBox()

    def getSpans(self):
        """Get the full `SpanSet`"""
        return self._footprint.getSpans()

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
            singles = self.singles[filterIndex]
        else:
            singles = []
            for f in filterIndex:
                singles.append(self.singles[f])
        result = MultibandFootprint(filters=filters, singles=singles)
        return result

    def getImage(self, bbox=None, fill=0):
        """Convert a `MultibandFootprint` to a `MultibandImage`

        Parameters
        ----------
        bbox: Box2I
            Bounding box of the resulting image.
            If no bounding box is specified, then the bounding box
            of the footprint is used.
        fill: float
            Value to use for any pixel in the resulting image
            outside of the `SpanSet`.

        Returns
        -------
        result: MultibandImage
            The resulting MultibandImage
        """
        if bbox is None:
            bbox = self.getBBox()
        images = []
        for heavy in self.singles:
            _img = Image(bbox, dtype=heavy.getImageArray().dtype)
            heavy.insert(_img)
            images.append(_img)
        image = MultibandImage(filters=self.filters, singles=images)
        return image

    def getArray(self, bbox=None, fill=0):
        """Convert a `MultibandFootprint` to a numpy array

        See `self.getImage` for a description of the parameters

        Results
        -------
        result: np.array
            Multiband image data cube
        """
        return self.getImage(bbox, fill).array

    def setBBox(self, bbox):
        """Overload the baseclass method to set the bounding box

        The bounding box of a `MultibandFootprint` is set, so we
        prevent the user from setting the bounding box in the base class.
        """
        raise ValueError("Cannot update the bounding box of a `MultibandFootprint")

    def setXY0(self, xy0):
        """Overload the baseclass method to set the XY0 position

        The bounding box of a `MultibandFootprint` is set, so we
        prevent the user from setting the bounding box or XY0 in the base class.
        """
        raise ValueError("Cannot update the bounding box of a `MultibandFootprint")

    def copy(self, deep=False):
        """Copy the current object

        Parameters
        ----------
        deep: bool
            Whether or not to make a deep copy
        """
        if deep:
            raise NotImplementedError("Cannot make a deep copy of a Heavy Footprint")
        else:
            result = MultibandFootprint(self.filters, self.singles)
        return result
