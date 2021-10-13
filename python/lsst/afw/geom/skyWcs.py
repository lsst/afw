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

import numpy as np
import scipy

from lsst.utils import continueClass
import lsst.geom
from ._python import reduceTransform
from ._geom import (SkyWcs, makeCdMatrix, makeFlippedWcs, makeModifiedWcs,
                    makeSkyWcs, makeTanSipWcs, makeWcsPairTransform,
                    getIntermediateWorldCoordsToSky, getPixelToIntermediateWorldCoords)
from ._hpxUtils import makeHpxWcs

__all__ = ["SkyWcs", "makeCdMatrix", "makeFlippedWcs", "makeSkyWcs",
           "makeModifiedWcs", "makeTanSipWcs", "makeWcsPairTransform",
           "getIntermediateWorldCoordsToSky", "getPixelToIntermediateWorldCoords",
           "makeHpxWcs"]


@continueClass
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

    def getRelativeRotationToWcs(self, otherWcs):
        """Get the difference in sky rotation angle to the specified wcs.

        Ignoring location on the sky, if another wcs were atop this one,
        what would the difference in rotation be? i.e. for

        otherWcs = createInitialSkyWcsFromBoresight(radec, rotation, detector)

        what is the value that needs to be added to ``self.rotation`` (or
        subtracted from `other.rotation``) to align them?

        Parameters
        ----------
        otherWcs : `lsst.afw.geom.SkyWcs`
            The wcs to calculate the angle to.

        Returns
        -------
        angle : `lsst.geom.Angle`
            The angle between this and the supplied wcs,
            over the half-open range [0, 2pi).
        """
        # Note: tests for this function live in
        # obs_lsst/tests/test_afwWcsUtil.py due to the need for an easy
        # constructor and instantiated detector, and the fact that afw
        # cannot depend on obs_base or obs_lsst.

        m1 = self.getCdMatrix()
        m2 = otherWcs.getCdMatrix()

        svd1 = scipy.linalg.svd(m1)
        svd2 = scipy.linalg.svd(m2)

        m1rot = np.matmul(svd1[0], svd1[2])
        m2rot = np.matmul(svd2[0], svd2[2])

        v_rot = [1, 0]

        v_rot = np.matmul(v_rot, m1rot)  # rotate by wcs1
        v_rot = np.matmul(v_rot, m2rot.T)  # rotate _back_ by wcs2

        rotation = np.arctan2(v_rot[1], v_rot[0])
        rotation = rotation % (2*np.pi)
        return lsst.geom.Angle(rotation, lsst.geom.radians)


SkyWcs.__reduce__ = reduceTransform
