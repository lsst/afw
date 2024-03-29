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

__all__ = ["DetectorBase", "DetectorTypeValNameDict", "DetectorTypeNameValDict"]

from lsst.utils import continueClass
from ._cameraGeom import DetectorBase, DetectorType

DetectorTypeValNameDict = {
    DetectorType.SCIENCE: "SCIENCE",
    DetectorType.FOCUS: "FOCUS",
    DetectorType.GUIDER: "GUIDER",
    DetectorType.WAVEFRONT: "WAVEFRONT",
}

DetectorTypeNameValDict = {val: key for key, val in
                           DetectorTypeValNameDict.items()}


@continueClass
class DetectorBase:  # noqa: F811
    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def fromConfig(self, config=None, numAmps=1):
        if config is not None:
            self.setSerial(config.serial)
            self.setType(DetectorType(config.detectorType))
            self.setPhysicalType(config.physicalType)
            self.setBBox(config.bbox)
            self.setPixelSize(config.pixelSize)
            self.setOrientation(config.orientation)
            self.setCrosstalk(config.getCrosstalk(numAmps))
