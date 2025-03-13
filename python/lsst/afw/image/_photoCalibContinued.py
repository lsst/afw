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

__all__ = ["PhotoCalib"]

import enum
import warnings

import numpy as np
from astropy import units

from lsst.utils import continueClass

from ._imageLib import PhotoCalib


class _UnsetEnum(enum.Enum):
    UNSET = enum.auto()


@continueClass
class PhotoCalib:  # noqa: F811
    def getLocalCalibrationArray(self, x, y):
        """Get the local calibration values (nJy/counts) for numpy arrays (pixels).

        Parameters
        ----------
        x : `np.ndarray` (N,)
            Array of x values (pixels).
        y : `np.ndarray` (N,)
            Array of y values (pixels).

        Returns
        -------
        localCalibration : `np.ndarray` (N,)
            Array of local calibration values (nJy/counts).
        """
        if self._isConstant:
            return np.full(len(x), self.getCalibrationMean())
        else:
            bf = self.computeScaledCalibration()
            return self.getCalibrationMean()*bf.evaluate(x, y)

    def instFluxToMagnitudeArray(self, instFluxes, x, y):
        """Convert instFlux (counts) to magnitudes for numpy arrays (pixels).

        Parameters
        ----------
        instFluxes : `np.ndarray` (N,)
            Array of instFluxes to convert (counts).
        x : `np.ndarray` (N,)
            Array of x values (pixels).
        y : `np.ndarray` (N,)
            Array of y values (pixels).

        Returns
        -------
        magnitudes : `astropy.units.Magnitude` (N,)
            Array of AB magnitudes.
        """
        scale = self.getLocalCalibrationArray(x, y)
        nanoJansky = (instFluxes*scale)*units.nJy

        return nanoJansky.to(units.ABmag)

    def magnitudeToInstFluxArray(self, magnitudes, x, y):
        """Convert magnitudes to instFlux (counts) for numpy arrays (pixels).

        Parameters
        ----------
        magnitudes : `np.ndarray` or `astropy.units.Magnitude` (N,)
            Array of AB magnitudes.
        x : `np.ndarray` (N,)
            Array of x values (pixels).
        y : `np.ndarray` (N,)
            Array of y values (pixels).

        Returns
        -------
        instFluxes : `np.ndarray` (N,)
            Array of instFluxes (counts).
        """
        scale = self.getLocalCalibrationArray(x, y)

        if not isinstance(magnitudes, units.Magnitude):
            _magnitudes = magnitudes*units.ABmag
        else:
            _magnitudes = magnitudes

        nanoJansky = _magnitudes.to(units.nJy).value

        return nanoJansky/scale

    # TODO[DM-49400]: remove this method and rename the pybind11 method to drop
    # the leading underscore.
    def calibrateImage(self, maskedImage, includeScaleUncertainty=_UnsetEnum.UNSET):
        """Return a flux calibrated image, with pixel values in nJy.

        Mask pixels are propagated directly from the input image.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            The masked image to calibrate.
        includeScaleUncertainty : `bool`, optional
             Deprecated and ignored; will be removed after v30.

        Returns
        ------
        calibrated : `lsst.afw.image.MaskedImage`
            The calibrated masked image.
        """
        if includeScaleUncertainty is not _UnsetEnum.UNSET:
            warnings.warn(
                "The 'includeScaleUncertainty' argument to calibrateImage is deprecated and does "
                "nothing.  It will be removed after v30.",
                category=FutureWarning
            )
        return self._calibrateImage(maskedImage)

    # TODO[DM-49400]: remove this method and rename the pybind11 method to drop
    # the leading underscore.
    def uncalibrateImage(self, maskedImage, includeScaleUncertainty=_UnsetEnum.UNSET):
        """Return a un-calibrated image, with pixel values in ADU (or whatever
        the original input to this photoCalib was).

        Mask pixels are propagated directly from the input image.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            The masked image with pixel units of nJy to uncalibrate.
        includeScaleUncertainty : `bool`, optional
            Deprecated and ignored; will be removed after v30.

        Returns
        uncalibrated : `lsst.afw.image.MaskedImage`
            The uncalibrated masked image.
        """
        if includeScaleUncertainty is not _UnsetEnum.UNSET:
            warnings.warn(
                "The 'includeScaleUncertainty' argument to uncalibrateImage is deprecated and does "
                "nothing.  It will be removed after v30.",
                category=FutureWarning
            )
        return self._uncalibrateImage(maskedImage)
