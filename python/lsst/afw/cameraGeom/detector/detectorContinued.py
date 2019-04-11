#
# LSST Data Management System
# Copyright 2016 LSST Corporation.
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

__all__ = ["Detector", "DetectorType",
           "SCIENCE", "FOCUS", "GUIDER", "WAVEFRONT"]

from lsst.utils import continueClass
from .detector import Detector, DetectorType

# export DetectorType enums as module globals for SWIG compatibility;
# @TODO update our code to stop using these globals and remove this code
SCIENCE = DetectorType.SCIENCE
FOCUS = DetectorType.FOCUS
GUIDER = DetectorType.GUIDER
WAVEFRONT = DetectorType.WAVEFRONT


@continueClass  # noqa: F811
class Detector:
    def __iter__(self):
        return (self[i] for i in range(len(self)))
