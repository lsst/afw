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

__all__ = ["ReadoutCornerValNameDict", "ReadoutCornerNameValDict"]

from lsst.utils import continueClass
from .amplifier import Amplifier, ReadoutCorner, AssemblyState


ReadoutCornerValNameDict = {
    ReadoutCorner.LL: "LL",
    ReadoutCorner.LR: "LR",
    ReadoutCorner.UR: "UR",
    ReadoutCorner.UL: "UL",
}
ReadoutCornerNameValDict = {val: key for key, val in
                            ReadoutCornerValNameDict.items()}


@continueClass  # noqa: F811
class Amplifier:
    def getDataBBox(self):
        myState = self.getAssemblyState()
        if myState == AssemblyState.SCIENCE:
            return self.getBBox()
        else:
            return self.getRawDataBBox()
