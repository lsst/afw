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

# Camera object below is the same one exported by the pybind11 camera
# module, so we don't need to re-export it here.
__all__ = []

from lsst.utils import continueClass, doImport
from .camera import Camera


@continueClass  # noqa: F811
class Camera:

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
