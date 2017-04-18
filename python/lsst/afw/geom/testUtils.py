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

__all__ = ["BoxGrid"]

from builtins import range
from builtins import object

import numpy as np

from .box import Box2I, Box2D


class BoxGrid(object):
    """!Divide a box into nx by ny sub-boxes that tile the region
    """

    def __init__(self, box, numColRow):
        """!Construct a BoxGrid

        The sub-boxes will be of the same type as `box` and will exactly tile `box`;
        they will also all be the same size, to the extent possible (some variation
        is inevitable for integer boxes that cannot be evenly divided.

        @param[in] box  box (an lsst.afw.geom.Box2I or Box2D);
                        the boxes in the grid will be of the same type
        @param[in] numColRow  number of columns and rows (a pair of ints)
        """
        if len(numColRow) != 2:
            raise RuntimeError(
                "numColRow=%r; must be a sequence of two integers" % (numColRow,))
        self._numColRow = tuple(int(val) for val in numColRow)

        if isinstance(box, Box2I):
            stopDelta = 1
        elif isinstance(box, Box2D):
            stopDelta = 0
        else:
            raise RuntimeError("Unknown class %s of box %s" % (type(box), box))
        self.boxClass = type(box)
        self.stopDelta = stopDelta

        minPoint = box.getMin()
        self.pointClass = type(minPoint)
        dtype = np.array(minPoint).dtype

        self._divList = [np.linspace(start=box.getMin()[i],
                                     stop=box.getMax()[i] + self.stopDelta,
                                     num=self._numColRow[i] + 1,
                                     endpoint=True,
                                     dtype=dtype) for i in range(2)]

    @property
    def numColRow(self):
        return self._numColRow

    def __getitem__(self, indXY):
        """!Return the box at the specified x,y index (a pair of ints)
        """
        beg = self.pointClass(*[self._divList[i][indXY[i]] for i in range(2)])
        end = self.pointClass(
            *[self._divList[i][indXY[i] + 1] - self.stopDelta for i in range(2)])
        return self.boxClass(beg, end)

    def __len__(self):
        return self.shape[0]*self.shape[1]

    def __iter__(self):
        """!Return an iterator over all boxes, where column varies most quickly
        """
        for row in range(self.numColRow[1]):
            for col in range(self.numColRow[0]):
                yield self[col, row]
