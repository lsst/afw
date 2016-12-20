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
from __future__ import absolute_import, division, print_function

from ._detector import Detector, DetectorType

__all__ = ["Detector", "DetectorType", "SCIENCE", "FOCUS", "GUIDER", "WAVEFRONT"]

# export DetectorType enums as module globals for SWIG compatibility;
# @TODO update our code to stop using these globals and remove this code
SCIENCE = DetectorType.SCIENCE
FOCUS = DetectorType.FOCUS
GUIDER = DetectorType.GUIDER
WAVEFRONT = DetectorType.WAVEFRONT


def __iter__(self):
    return (self[i] for i in range(len(self)))

Detector.__iter__ = __iter__
