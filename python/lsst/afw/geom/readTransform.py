#
# LSST Data Management System
# Copyright 2017 LSST/AURA.
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

import astshim

from .python import transformRegistry

__all__ = ["readTransform"]


def readTransform(path):
    """Read a Transform from a file that was saved using Transform method `saveToFile`
    """
    with open(path, "r") as inFile:
        transformClassName = inFile.readline().strip()
        try:
            transformClass = transformRegistry[transformClassName]
        except LookupError:
            raise RuntimeError("Unknown transform class %r specified in file %r" %
                               (transformClassName, path))

        frameSetStr = inFile.read()
        stream = astshim.StringStream(frameSetStr)
        frameSet = astshim.Channel(stream).read()
        if not isinstance(frameSet, astshim.FrameSet):
            raise RuntimeError("Found astshim object of type %s instead of FrameSet in file %r" %
                               (type(frameSet).__name, path))
        return transformClass(frameSet)
