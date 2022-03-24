# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["Exposure"]

from lsst.utils.deprecated import deprecate_pybind11
import numpy as np

from lsst.utils import TemplateMeta

from ..image._slicing import supportSlicing
from ..image._disableArithmetic import disableImageArithmetic
from ..image._fitsIoWithOptions import imageReadFitsWithOptions, exposureWriteFitsWithOptions
from ._exposure import ExposureI, ExposureF, ExposureD, ExposureU, ExposureL
from .exposureUtils import bbox_to_convex_polygon, bbox_contains_sky_coords
from lsst.pex.exceptions import InvalidParameterError
import lsst.geom as geom

SIGMA_2_FWHM = 2.*np.sqrt(2.*np.log(2.))


class Exposure(metaclass=TemplateMeta):

    def _set(self, index, value, origin):
        """Set the pixel at the given index to a triple (value, mask, variance).

        Parameters
        ----------
        index : `geom.Point2I`
            Position of the pixel to assign to.
        value : `tuple`
            A tuple of (value, mask, variance) scalars.
        origin : `ImageOrigin`
            Coordinate system of ``index`` (`PARENT` or `LOCAL`).
        """
        self.maskedImage._set(index, value=value, origin=origin)

    def _get(self, index, origin):
        """Return a triple (value, mask, variance) at the given index.

        Parameters
        ----------
        index : `geom.Point2I`
            Position of the pixel to assign to.
        origin : `ImageOrigin`
            Coordinate system of ``index`` (`PARENT` or `LOCAL`).
        """
        return self.maskedImage._get(index, origin=origin)

    def __reduce__(self):
        from lsst.afw.fits import reduceToFits
        return reduceToFits(self)

    def convertF(self):
        return ExposureF(self, deep=True)

    def convertD(self):
        return ExposureD(self, deep=True)

    def getImage(self):
        return self.maskedImage.image

    def setImage(self, image):
        self.maskedImage.image = image

    image = property(getImage, setImage)

    def getMask(self):
        return self.maskedImage.mask

    def setMask(self, mask):
        self.maskedImage.mask = mask

    mask = property(getMask, setMask)

    def getVariance(self):
        return self.maskedImage.variance

    def setVariance(self, variance):
        self.maskedImage.variance = variance

    variance = property(getVariance, setVariance)

    def getConvexPolygon(self, padding=10):
        """Get the convex polygon associated with the bounding box corners.

        The returned polygon has additional padding to ensure that the
        bounding box is entirely contained within it.  To ensure a set
        of coordinates are entirely contained within an exposure, run
        ``exposure.containsSkyCoords()``.  The default padding
        size was chosen to be sufficient for the most warped detectors at
        the edges of the HyperSuprimeCam focal plane.

        Parameters
        ----------
        padding : `int`
            Pixel padding to ensure that bounding box is entirely contained
            within the resulting polygon.

        Returns
        -------
        convexPolygon : `lsst.sphgeom.ConvexPolygon`
            Returns `None` if exposure does not have a valid WCS.
        """
        if self.wcs is None:
            return None

        return bbox_to_convex_polygon(self.getBBox(), self.wcs, padding=padding)

    convex_polygon = property(getConvexPolygon)

    def containsSkyCoords(self, ra, dec, padding=10):
        """Check if a set of sky positions is in the pixel bounding box.

        The default padding size was chosen to be sufficient for the
        most warped detectors at the edges of the HyperSuprimeCam focal plane.

        Parameters
        ----------
        ra : `astropy.Quantity`, (N,)
            Array of Right Ascension, angular units.
        dec : `astropy.Quantity`, (N,)
            Array of Declination, angular units.
        padding : `int`, optional
            Pixel padding to ensure that bounding box is entirely contained
            within the sky polygon (see ``getConvexPolygon()``).

        Returns
        -------
        contained : `np.ndarray`, (N,)
            Boolean array indicating which points are contained in the
            bounding box.

        Raises
        ------
        ValueError if exposure does not have a valid wcs.
        """
        if self.wcs is None:
            raise ValueError("Exposure does not have a valid WCS.")

        return bbox_contains_sky_coords(
            self.getBBox(),
            self.wcs,
            ra,
            dec,
            padding=padding)

    readFitsWithOptions = classmethod(imageReadFitsWithOptions)

    writeFitsWithOptions = exposureWriteFitsWithOptions

    def evaluateMedianFwhm(self, fwhmExposureBuffer=0.05, fwhmExposureGrid=10):
        """Get the median PSF FWHM by evaluating it on a grid of points on an exposure

        Parameters
        ----------
        fwhmExposureBuffer : `float`, optional
            Fractional buffer margin to be left out of all sides of the image during construction
            of grid to compute median PSF FWHM in an exposure
        fwhmExposureGrid : `int`, optional
            Grid size to compute the median FWHM in an exposure

        Returns
        -------
        medianFwhm : `float`
            The median PSF FWHM on the exposure

        """

        psf = self.getPsf()
        bbox = self.getBBox()

        xmax, ymax = bbox.getMax()
        xmin, ymin = bbox.getMin()

        xbuffer = fwhmExposureBuffer*(xmax-xmin)
        ybuffer = fwhmExposureBuffer*(ymax-ymin)

        xx, yy = np.meshgrid(np.linspace(xmin+xbuffer, xmax-xbuffer, fwhmExposureGrid),
                             np.linspace(ymin+ybuffer, ymax-ybuffer, fwhmExposureGrid))

        sigPixArray = np.zeros(xx.size)
        for ii, (x, y) in enumerate(zip(xx.flat, yy.flat)):
            try:
                sigPixArray[ii] = psf.computeShape(geom.Point2D(x, y)).getDeterminantRadius()
            except InvalidParameterError:
                sigPixArray[ii] = np.nan

        sigPix = np.nanmedian(sigPixArray)

        return sigPix*SIGMA_2_FWHM


Exposure.register(np.int32, ExposureI)
Exposure.register(np.float32, ExposureF)
Exposure.register(np.float64, ExposureD)
Exposure.register(np.uint16, ExposureU)
Exposure.register(np.uint64, ExposureL)
Exposure.alias("I", ExposureI)
Exposure.alias("F", ExposureF)
Exposure.alias("D", ExposureD)
Exposure.alias("U", ExposureU)
Exposure.alias("L", ExposureL)

for cls in set(Exposure.values()):
    supportSlicing(cls)
    disableImageArithmetic(cls)

    cls.getFilter = deprecate_pybind11(
        cls.getFilter,
        reason="Replaced by getFilterLabel. Will be removed after v22.",
        version="v22.0")

    cls.setFilter = deprecate_pybind11(
        cls.setFilter,
        reason="Replaced by setFilterLabel. Will be removed after v22.",
        version="v22.0")
