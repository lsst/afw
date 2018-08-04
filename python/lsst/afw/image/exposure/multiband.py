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

__all__ = ["MultibandExposure"]

import numpy as np

from lsst.geom import Point2D
from . import Exposure, ExposureF
from ..image.multiband import MultibandImage, MultibandTripleBase
from ..image.multiband import tripleFromSingles, tripleFromArrays, makeTripleFromKwargs
from ..maskedImage import MaskedImage


class MultibandExposure(MultibandTripleBase):
    """MultibandExposure class

    This class acts as a container for multiple `afw.Exposure` objects.
    All exposures must have the same bounding box, and the associated
    images must all have the same data type.

    See `MultibandTripleBase` for parameter definitions.
    """
    def __init__(self, filters, image, mask, variance, psfs=None):
        super().__init__(filters, image, mask, variance)
        if psfs is not None:
            for psf, exposure in zip(psfs, self.singles):
                exposure.setPsf(psf)

    @staticmethod
    def fromExposures(filters, singles):
        """Construct a MultibandImage from a collection of single band images

        see `tripleFromExposures` for a description of parameters
        """
        psfs = [s.getPsf() for s in singles]
        return tripleFromSingles(MultibandExposure, filters, singles, psfs=psfs)

    @staticmethod
    def fromArrays(filters, image, mask, variance, bbox=None):
        """Construct a MultibandExposure from a collection of arrays

        see `tripleFromArrays` for a description of parameters
        """
        return tripleFromArrays(MultibandExposure, filters, image, mask, variance, bbox)

    @staticmethod
    def fromKwargs(filters, filterKwargs, singleType=ExposureF, **kwargs):
        """Build a MultibandImage from a set of keyword arguments

        see `makeTripleFromKwargs` for a description of parameters
        """
        return makeTripleFromKwargs(MultibandExposure, filters, filterKwargs, singleType, **kwargs)

    def _buildSingles(self, image=None, mask=None, variance=None):
        """Make a new list of single band objects

        Parameters
        ----------
        image: `list`
            List of `Image` objects that represent the image in each band.
        mask: `list`
            List of `Mask` objects that represent the mask in each band.
        variance: `list`
            List of `Image` objects that represent the variance in each band.

        Returns
        -------
        singles: tuple
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
        for f in self.filters:
            maskedImage = MaskedImage(image=image[f], mask=mask[f], variance=variance[f], dtype=dtype)
            single = Exposure(maskedImage, dtype=dtype)
            singles.append(single)
        return tuple(singles)

    @staticmethod
    def fromButler(butler, filters, filterKwargs, *args, **kwargs):
        """Load a multiband exposure from a butler

        Because each band is stored in a separate exposure file,
        this method can be used to load all of the exposures for
        a given set of bands

        Parameters
        ----------
        butler: `Butler`
            Butler connection to use to load the single band
            calibrated images
        filters: `list` or `str`
            List of filter names for each band
        filterKwargs: `dict`
            Keyword arguments to pass to the Butler
            that are different for each filter.
            The keys are the names of the arguments and the values
            should also be dictionaries, with filter names as keys
            and the value of the argument for the given filter as values.
        args: `list`
            Arguments to the Butler.
        kwargs: `dict`
            Keyword arguments to pass to the Butler
            that are the same in all bands.

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
        return MultibandExposure.fromExposures(filters, exposures)

    def computePsfKernelImage(self, position=None):
        """Get a multiband PSF image

        The PSF Kernel Image is computed for each band
        and combined into a (filter, y, x) array and stored
        as `self._psfImage`.
        The result is not cached, so if the same PSF is expected
        to be used multiple times it is a good idea to store the
        result in another variable.

        Parameters
        ----------
        position: `Point2D` or `tuple`
            Coordinates to evaluate the PSF. If `position` is `None`
            then `Psf.getAveragePosition()` is used.

        Returns
        -------
        self._psfImage: array
            The multiband PSF image.
        """
        psfs = []
        # Make the coordinates into a Point2D (if necessary)
        if not isinstance(position, Point2D) and position is not None:
            position = Point2D(position[0], position[1])
        for single in self.singles:
            if position is None:
                psfs.append(single.getPsf().computeKernelImage().array)
            else:
                psfs.append(single.getPsf().computeKernelImage(position).array)
        psfs = np.array(psfs)
        psfImage = MultibandImage(self.filters, array=psfs)
        return psfImage

    def computePsfImage(self, position=None):
        """Get a multiband PSF image

        The PSF Kernel Image is computed for each band
        and combined into a (filter, y, x) array and stored
        as `self._psfImage`.
        The result is not cached, so if the same PSF is expected
        to be used multiple times it is a good idea to store the
        result in another variable.

        Parameters
        ----------
        position: `Point2D` or `tuple`
            Coordinates to evaluate the PSF. If `position` is `None`
            then `Psf.getAveragePosition()` is used.

        Returns
        -------
        self._psfImage: array
            The multiband PSF image.
        """
        psfs = []
        # Make the coordinates into a Point2D (if necessary)
        if not isinstance(position, Point2D) and position is not None:
            position = Point2D(position[0], position[1])
        for single in self.singles:
            if position is None:
                psfs.append(single.getPsf().computeImage().array)
            else:
                psfs.append(single.getPsf().computeImage(position).array)
        psfs = np.array(psfs)
        psfImage = MultibandImage(self.filters, array=psfs)
        return psfImage
