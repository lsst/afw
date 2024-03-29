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

__all__ = ['Camera']

import numpy as np
from lsst.utils import continueClass, doImport
from ._cameraGeom import Camera, FOCAL_PLANE


@continueClass
class Camera:  # noqa: F811

    def getPupilFactory(self, visitInfo, pupilSize, npix, **kwargs):
        """Construct a PupilFactory.

        Parameters
        ----------
        visitInfo : `~lsst.afw.image.VisitInfo`
            VisitInfo object for a particular exposure.
        pupilSize : `float`
            Size in meters of constructed Pupil array. Note that this may be
            larger than the actual diameter of the illuminated pupil to
            accommodate zero-padding.
        npix : `int`
            Constructed Pupils will be npix x npix.
        **kwargs : `dict`
            Other keyword arguments forwarded to the PupilFactoryClass
            constructor.
        """
        cls = doImport(self.getPupilFactoryName())
        return cls(visitInfo, pupilSize, npix, **kwargs)

    @property
    def telescopeDiameter(self):
        cls = doImport(self.getPupilFactoryName())
        return cls.telescopeDiameter

    def computeMaxFocalPlaneRadius(self):
        """Compute the maximum radius on the focal plane of the corners of all
        detectors in this camera.

        Returns
        -------
        focalRadius : `float`
            Maximum focal plane radius in FOCAL_PLANE units (mm).
        """
        radii = []
        for detector in self:
            for corner in detector.getCorners(FOCAL_PLANE):
                radii.append(np.hypot(*corner))
        return np.max(radii)
