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


"""Application Framework classes to handle a mosaic camera's geometry
"""
from ._cameraGeom import *
from ._detector import *
from ._amplifier import *
from .cameraConfig import *
from ._detectorCollection import *
from ._camera import *
from ._cameraFactory import *
from ._cameraGeomEnumDicts import *
from ._makePixelToTanPixel import *
from ._assembleImage import *
from ._rotateBBoxBy90 import *
from .pupil import *
from ._transformConfig import *
NullLinearityType = "None"  # linearity type indicating no linearity correction wanted
