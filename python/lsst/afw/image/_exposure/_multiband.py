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

__all__ = ["MultibandExposure", "computePsfImage", "IncompleteDataError"]

import numpy as np

from lsst.geom import Point2D, Point2I, Box2I
from lsst.pex.exceptions import InvalidParameterError
from . import Exposure, ExposureF
from ..utils import projectImage
from .._image._multiband import MultibandTripleBase, MultibandPixel, MultibandImage
from .._image._multiband import tripleFromSingles, tripleFromArrays, makeTripleFromKwargs
from .._maskedImage import MaskedImage


class IncompleteDataError(Exception):
    """The PSF could not be computed due to incomplete data

    Attributes
    ----------
    missingBands: `list[str]`
        The bands for which the PSF could not be calculated.
    position: `Point2D`
        The point at which the PSF could not be calcualted in the
        missing bands.
    partialPsf: `MultibandImage`
        The image of the PSF using only the bands that successfully
        computed a PSF image.

    Parameters
    ----------
    bands : `list` of `str`
        The full list of bands in the `MultibandExposure` generating
        the PSF.
    """
    def __init__(self, bands, position, partialPsf):
        missingBands = [band for band in bands if band not in partialPsf.bands]

        self.missingBands = missingBands
        self.position = position
        self.partialPsf = partialPsf
        message = f"Failed to compute PSF at {position} in {missingBands}"
        super().__init__(message)


def computePsfImage(psfModels, position, useKernelImage=True):
    """Get a multiband PSF image

    The PSF Image or PSF Kernel Image is computed for each band
    and combined into a (band, y, x) array.

    Parameters
    ----------
    psfModels : `dict[str, lsst.afw.detection.Psf]`
        The list of PSFs in each band.
    position : `Point2D` or `tuple`
        Coordinates to evaluate the PSF.
    useKernelImage: `bool`
        Execute ``Psf.computeKernelImage`` when ``True`,
        ``PSF/computeImage`` when ``False``.

    Returns
    -------
    psfs: `lsst.afw.image.MultibandImage`
        The multiband PSF image.
    """
    psfs = {}
    # Make the coordinates into a Point2D (if necessary)
    if not isinstance(position, Point2D):
        position = Point2D(position[0], position[1])

    incomplete = False

    for band, psfModel in psfModels.items():
        try:
            if useKernelImage:
                psf = psfModel.computeKernelImage(position)
            else:
                psf = psfModel.computeImage(position)
            psfs[band] = psf
        except InvalidParameterError:
            incomplete = True

    left = np.min([psf.getBBox().getMinX() for psf in psfs.values()])
    bottom = np.min([psf.getBBox().getMinY() for psf in psfs.values()])
    right = np.max([psf.getBBox().getMaxX() for psf in psfs.values()])
    top = np.max([psf.getBBox().getMaxY() for psf in psfs.values()])
    bbox = Box2I(Point2I(left, bottom), Point2I(right, top))

    psf_images = [projectImage(psf, bbox) for psf in psfs.values()]

    mPsf = MultibandImage.fromImages(list(psfs.keys()), psf_images)

    if incomplete:
        raise IncompleteDataError(list(psfModels.keys()), position, mPsf)

    return mPsf


class MultibandExposure(MultibandTripleBase):
    """MultibandExposure class

    This class acts as a container for multiple `afw.Exposure` objects.
    All exposures must have the same bounding box, and the associated
    images must all have the same data type.

    See `MultibandTripleBase` for parameter definitions.
    """
    def __init__(self, bands, image, mask, variance, psfs=None):
        super().__init__(bands, image, mask, variance)
        if psfs is not None:
            for psf, exposure in zip(psfs, self.singles):
                exposure.setPsf(psf)

    @staticmethod
    def fromExposures(bands, singles):
        """Construct a MultibandImage from a collection of single band images

        see `tripleFromExposures` for a description of parameters
        """
        psfs = [s.getPsf() for s in singles]
        return tripleFromSingles(MultibandExposure, bands, singles, psfs=psfs)

    @staticmethod
    def fromArrays(bands, image, mask, variance, bbox=None):
        """Construct a MultibandExposure from a collection of arrays

        see `tripleFromArrays` for a description of parameters
        """
        return tripleFromArrays(MultibandExposure, bands, image, mask, variance, bbox)

    @staticmethod
    def fromKwargs(bands, bandKwargs, singleType=ExposureF, **kwargs):
        """Build a MultibandImage from a set of keyword arguments

        see `makeTripleFromKwargs` for a description of parameters
        """
        return makeTripleFromKwargs(MultibandExposure, bands, bandKwargs, singleType, **kwargs)

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
        for f in self.bands:
            maskedImage = MaskedImage(image=image[f], mask=mask[f], variance=variance[f], dtype=dtype)
            single = Exposure(maskedImage, dtype=dtype)
            singles.append(single)
        return tuple(singles)

    @staticmethod
    def fromButler(butler, bands, *args, **kwargs):
        """Load a multiband exposure from a butler

        Because each band is stored in a separate exposure file,
        this method can be used to load all of the exposures for
        a given set of bands

        Parameters
        ----------
        butler: `lsst.daf.butler.Butler`
            Butler connection to use to load the single band
            calibrated images
        bands: `list` or `str`
            List of names for each band
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
        for band in bands:
            exposures.append(butler.get(*args, band=band, **kwargs))
        return MultibandExposure.fromExposures(bands, exposures)

    def computePsfKernelImage(self, position):
        """Get a multiband PSF image

        The PSF Kernel Image is computed for each band
        and combined into a (band, y, x) array and stored
        as `self._psfImage`.
        The result is not cached, so if the same PSF is expected
        to be used multiple times it is a good idea to store the
        result in another variable.

        Parameters
        ----------
        position: `Point2D` or `tuple`
            Coordinates to evaluate the PSF.

        Returns
        -------
        self._psfImage: array
            The multiband PSF image.
        """
        return computePsfImage(
            psfModels=self.getPsfs(),
            position=position,
            useKernelImage=True,
        )

    def computePsfImage(self, position=None):
        """Get a multiband PSF image

        The PSF Kernel Image is computed for each band
        and combined into a (band, y, x) array and stored
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
        return computePsfImage(
            psfModels=self.getPsfs(),
            position=position,
            useKernelImage=True,
        )

    def getPsfs(self):
        """Extract the PSF model in each band

        Returns
        -------
        psfs : `dict` of `lsst.afw.detection.Psf`
            The PSF in each band
        """
        return {band: self[band].getPsf() for band in self.bands}

    def _slice(self, bands, bandIndex, indices):
        """Slice the current object and return the result

        See `Multiband._slice` for a list of the parameters.
        This overwrites the base method to attach the PSF to
        each individual exposure.
        """
        image = self.image._slice(bands, bandIndex, indices)
        if self.mask is not None:
            mask = self._mask._slice(bands, bandIndex, indices)
        else:
            mask = None
        if self.variance is not None:
            variance = self._variance._slice(bands, bandIndex, indices)
        else:
            variance = None

        # If only a single pixel is selected, return the tuple of MultibandPixels
        if isinstance(image, MultibandPixel):
            if mask is not None:
                assert isinstance(mask, MultibandPixel)
            if variance is not None:
                assert isinstance(variance, MultibandPixel)
            return (image, mask, variance)

        _psfs = self.getPsfs()
        psfs = [_psfs[band] for band in bands]

        result = MultibandExposure(
            bands=bands,
            image=image,
            mask=mask,
            variance=variance,
            psfs=psfs,
        )

        assert all([r.getBBox() == result._bbox for r in [result._mask, result._variance]])
        return result
