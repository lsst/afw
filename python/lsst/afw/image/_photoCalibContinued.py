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

import numpy as np
from astropy import units

from lsst.utils import continueClass

from ._imageLib import PhotoCalib


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
