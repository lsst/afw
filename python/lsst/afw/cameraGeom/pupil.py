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

__all__ = ['PupilFactory', 'Pupil']

import numpy as np


class Pupil:
    """Pupil obscuration function.

    Parameters
    ----------
    illuminated : `numpy.ndarray`, (Nx, Ny)
        2D numpy array indicating which parts of the pupil plane are
        illuminated.
    size : `float`
        Size of pupil plane array in meters. Note that this may be larger
        than the actual diameter of the illuminated pupil to accommodate
        zero-padding.
    scale : `float`
        Sampling interval of pupil plane array in meters.
    """

    def __init__(self, illuminated, size, scale):
        self.illuminated = illuminated
        self.size = size
        self.scale = scale


class PupilFactory:
    """Pupil obscuration function factory for use with Fourier optics.

    Parameters
    ----------
    visitInfo : `lsst.afw.image.VisitInfo`
        Visit information for a particular exposure.
    pupilSize : `float`
        Size in meters of constructed Pupil array.
        Note that this may be larger than the actual diameter of the
        illuminated pupil to accommodate zero-padding.
    npix : `int`
        Constructed Pupils will be npix x npix.
    """

    def __init__(self, visitInfo, pupilSize, npix):
        self.visitInfo = visitInfo
        self.pupilSize = pupilSize
        self.npix = npix
        self.pupilScale = pupilSize/npix
        u = (np.arange(npix, dtype=np.float64) - (npix - 1)/2) * self.pupilScale
        self.u, self.v = np.meshgrid(u, u)

    def getPupil(self, point):
        """Calculate a Pupil at a given point in the focal plane.

        Parameters
        ----------
        point : `lsst.geom.Point2D`
          The focal plane coordinates.

        Returns
        -------
        pupil : `Pupil`
            The Pupil at ``point``.
        """
        raise NotImplementedError(
            "PupilFactory not implemented for this camera")

    @staticmethod
    def _pointLineDistance(p0, p1, p2):
        """Compute the right-angle distance between the points given by `p0`
        and the line that passes through `p1` and `p2`.

        Parameters
        ----------
        p0 : `tuple` of `numpy.ndarray`
            2-tuple of numpy arrays (x, y focal plane coordinates)
        p1 : ``pair`` of `float`
            x,y focal plane coordinates
        p2 : ``pair`` of `float`
            x,y focal plane coordinates

        Returns
        -------
        distances : `numpy.ndarray`
            Numpy array of distances; shape congruent to p0[0].
        """
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2
        dy21 = y2 - y1
        dx21 = x2 - x1
        return np.abs(dy21*x0 - dx21*y0 + x2*y1 - y2*x1)/np.hypot(dy21, dx21)

    def _fullPupil(self):
        """Make a fully-illuminated Pupil.

        Returns
        -------
        pupil : `Pupil`
            The illuminated pupil.
        """
        illuminated = np.ones(self.u.shape, dtype=np.bool)
        return Pupil(illuminated, self.pupilSize, self.pupilScale)

    def _cutCircleInterior(self, pupil, p0, r):
        """Cut out the interior of a circular region from a Pupil.

        Parameters
        ----------
        pupil : `Pupil`
            Pupil to modify in place.
        p0 : `pair`` of `float`
            2-tuple indicating region center.
        r : `float`
            Circular region radius.
        """
        r2 = (self.u - p0[0])**2 + (self.v - p0[1])**2
        pupil.illuminated[r2 < r**2] = False

    def _cutCircleExterior(self, pupil, p0, r):
        """Cut out the exterior of a circular region from a Pupil.

        Parameters
        ----------
        pupil : `Pupil`
            Pupil to modify in place
        p0 : `pair`` of `float`
            2-tuple indicating region center.
        r : `float`
            Circular region radius.
        """
        r2 = (self.u - p0[0])**2 + (self.v - p0[1])**2
        pupil.illuminated[r2 > r**2] = False

    def _cutRay(self, pupil, p0, angle, thickness):
        """Cut out a ray from a Pupil.

        Parameters
        ----------
        pupil : `Pupil`
            Pupil to modify in place.
        p0 : `pair`` of `float`
            2-tuple indicating ray starting point.
        angle : `pair` of `float`
            Ray angle measured CCW from +x.
        thickness : `float`
            Thickness of cutout.
        """
        angleRad = angle.asRadians()
        # the 1 is arbitrary, just need something to define another point on
        # the line
        p1 = (p0[0] + 1, p0[1] + np.tan(angleRad))
        d = PupilFactory._pointLineDistance((self.u, self.v), p0, p1)
        pupil.illuminated[(d < 0.5*thickness) &
                          ((self.u - p0[0])*np.cos(angleRad) +
                           (self.v - p0[1])*np.sin(angleRad) >= 0)] = False

    def _centerPupil(self, pupil):
        """Center the illuminated portion of the pupil in array.

        Parameters
        ----------
        pupil : `Pupil`
            Pupil to modify in place
        """
        def center(arr, axis):
            smash = np.sum(arr, axis=axis)
            w = np.where(smash)[0]
            return int(0.5*(np.min(w)+np.max(w)))
        ycenter = center(pupil.illuminated, 0)
        xcenter = center(pupil.illuminated, 1)
        ytarget = pupil.illuminated.shape[0]//2
        xtarget = pupil.illuminated.shape[1]//2
        pupil.illuminated = np.roll(np.roll(pupil.illuminated,
                                            xtarget-xcenter,
                                            axis=0),
                                    ytarget-ycenter,
                                    axis=1)
