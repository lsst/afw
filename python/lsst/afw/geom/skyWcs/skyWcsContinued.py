#
# LSST Data Management System
# Copyright 2017 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import numpy as np

from lsst.utils import continueClass
from ..python import reduceTransform
from .skyWcs import SkyWcs

__all__ = []


@continueClass  # noqa: F811 (FIXME: remove for py 3.8+)
class SkyWcs:  # noqa: F811
    def pixelToSkyArray(self, x, y, degrees=False):
        """
        Convert numpy array pixels (x, y) to numpy array sky (ra, dec)
        positions.

        Parameters
        ----------
        x : `np.ndarray`
            Array of x values.
        y : `np.ndarray`
            Array of y values.
        degrees : `bool`, optional
            Return ra, dec arrays in degrees if True.

        Returns
        -------
        ra : `np.ndarray`
            Array of Right Ascension.  Units are radians unless
            degrees=True.
        dec : `np.ndarray`
            Array of Declination.  Units are radians unless
            degrees=True.
        """
        xy = np.vstack((x, y))
        ra, dec = np.vsplit(self.getTransform().getMapping().applyForward(xy), 2)
        ra %= (2.*np.pi)

        if degrees:
            return np.rad2deg(ra.ravel()), np.rad2deg(dec.ravel())
        else:
            return ra.ravel(), dec.ravel()

    def skyToPixelArray(self, ra, dec, degrees=False):
        """
        Convert numpy array sky (ra, dec) positions to numpy array
        pixels (x, y).

        Parameters
        ----------
        ra : `np.ndarray`
            Array of Right Ascension.  Units are radians unless
            degrees=True.
        dec : `np.ndarray`
            Array of Declination.  Units are radians unless
            degrees=True.
        degrees : `bool`, optional
            Input ra, dec arrays are degrees if True.

        Returns
        -------
        x : `np.ndarray`
            Array of x values.
        y : `np.ndarray`
            Array of y values.
        """
        radec = np.vstack((ra, dec))
        if degrees:
            radec = np.deg2rad(radec)

        x, y = np.vsplit(self.getTransform().getMapping().applyInverse(radec), 2)

        return x.ravel(), y.ravel()


SkyWcs.__reduce__ = reduceTransform
