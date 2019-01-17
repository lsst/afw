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

"""
Tests for statisticsStack row/column statistics

Run with:
   python test_rowColumnStats.py
or
   pytest test_rowColumnStats.py
"""
import unittest

import lsst.geom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.utils.tests


class RowColumnStatisticsTestCase(unittest.TestCase):

    def setUp(self):

        # fill an image with a gradient
        self.n = 8
        self.img = afwImage.ImageF(lsst.geom.Extent2I(self.n, self.n), 0)

        # these are the known answers for comparison
        def nVector(n, v):
            return [v for i in range(n)]
        self.column = nVector(self.n, 0.0)
        self.row = nVector(self.n, 0.0)
        self.colPlus = nVector(self.n, 0.0)
        self.colMinus = nVector(self.n, 0.0)
        self.colMult = nVector(self.n, 0.0)
        self.colDiv = nVector(self.n, 0.0)
        self.rowPlus = nVector(self.n, 0.0)
        self.rowMinus = nVector(self.n, 0.0)
        self.rowMult = nVector(self.n, 0.0)
        self.rowDiv = nVector(self.n, 0.0)

        # set the values in the image, and keep track of the stats to verify
        # things
        for y in range(self.n):
            for x in range(self.n):
                val = 1.0*x + 2.0*y
                self.img[x, y, afwImage.LOCAL] = val
                self.column[y] += val
                self.row[x] += val

        for i in range(self.n):
            self.row[i] /= self.n
            self.column[i] /= self.n
            self.colPlus[i] = self.img[0, i, afwImage.LOCAL] + self.column[i]

        # get stats on the columns and rows
        self.imgProjectCol = afwMath.statisticsStack(
            self.img, afwMath.MEAN, 'x')
        self.imgProjectRow = afwMath.statisticsStack(
            self.img, afwMath.MEAN, 'y')

    def tearDown(self):
        del self.img
        del self.imgProjectCol
        del self.imgProjectRow

    def testColumnStats(self):
        """Test the column statistics """
        for i in range(self.n):
            self.assertEqual(self.imgProjectCol[0, i, afwImage.LOCAL][0], self.column[i])

    def testRowStats(self):
        """Test the row statistics """
        for i in range(self.n):
            self.assertEqual(self.imgProjectRow[i, 0, afwImage.LOCAL][0], self.row[i])

    def testColumnOperators(self):
        """ Test operator overloading on columns """

        columnSlice = afwImage.ImageSliceF(self.imgProjectCol.getImage())

        imgAdd = self.img + columnSlice
        imgAdd2 = columnSlice + self.img
        imgSub = self.img - columnSlice
        imgMul = self.img * columnSlice
        imgMul2 = columnSlice * self.img
        imgDiv = self.img / columnSlice

        for i in range(self.n):
            self.assertAlmostEqual(imgAdd[0, i, afwImage.LOCAL],
                                   self.img[0, i, afwImage.LOCAL] + columnSlice[0, i, afwImage.LOCAL])
            self.assertAlmostEqual(imgAdd2[0, i, afwImage.LOCAL],
                                   imgAdd[0, i, afwImage.LOCAL])
            self.assertAlmostEqual(imgSub[0, i, afwImage.LOCAL],
                                   self.img[0, i, afwImage.LOCAL] - columnSlice[0, i, afwImage.LOCAL])
            self.assertAlmostEqual(imgMul[0, i, afwImage.LOCAL],
                                   self.img[0, i, afwImage.LOCAL] * columnSlice[0, i, afwImage.LOCAL])
            self.assertAlmostEqual(imgMul2[0, i, afwImage.LOCAL], imgMul[0, i, afwImage.LOCAL])
            self.assertAlmostEqual(imgDiv[0, i, afwImage.LOCAL],
                                   self.img[0, i, afwImage.LOCAL] / columnSlice[0, i, afwImage.LOCAL])


#################################################################
# Test suite boiler plate
#################################################################
class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
