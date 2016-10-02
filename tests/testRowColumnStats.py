#pybind11##!/usr/bin/env python
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see <http://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#
#pybind11## -*- python -*-
#pybind11#"""
#pybind11#Tests for statisticsStack row/column statistics
#pybind11#
#pybind11#Run with:
#pybind11#   ./rowColumnStats.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import rowColumnStats; rowColumnStats.run()
#pybind11#"""
#pybind11#from __future__ import division
#pybind11#from builtins import range
#pybind11#
#pybind11###########################
#pybind11## rowColumnStats.py
#pybind11## Steve Bickerton
#pybind11## An python test to check the row/column statistics from statisticsStack
#pybind11#
#pybind11#import unittest
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.utils.tests
#pybind11#
#pybind11#
#pybind11#class RowColumnStatisticsTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#
#pybind11#        # fill an image with a gradient
#pybind11#        self.n = 8
#pybind11#        self.img = afwImage.ImageF(afwGeom.Extent2I(self.n, self.n), 0)
#pybind11#
#pybind11#        # these are the known answers for comparison
#pybind11#        def nVector(n, v):
#pybind11#            return [v for i in range(n)]
#pybind11#        self.column = nVector(self.n, 0.0)
#pybind11#        self.row = nVector(self.n, 0.0)
#pybind11#        self.colPlus = nVector(self.n, 0.0)
#pybind11#        self.colMinus = nVector(self.n, 0.0)
#pybind11#        self.colMult = nVector(self.n, 0.0)
#pybind11#        self.colDiv = nVector(self.n, 0.0)
#pybind11#        self.rowPlus = nVector(self.n, 0.0)
#pybind11#        self.rowMinus = nVector(self.n, 0.0)
#pybind11#        self.rowMult = nVector(self.n, 0.0)
#pybind11#        self.rowDiv = nVector(self.n, 0.0)
#pybind11#
#pybind11#        # set the values in the image, and keep track of the stats to verify things
#pybind11#        for y in range(self.n):
#pybind11#            for x in range(self.n):
#pybind11#                val = 1.0*x + 2.0*y
#pybind11#                self.img.set(x, y, val)
#pybind11#                self.column[y] += val
#pybind11#                self.row[x] += val
#pybind11#
#pybind11#        for i in range(self.n):
#pybind11#            self.row[i] /= self.n
#pybind11#            self.column[i] /= self.n
#pybind11#            self.colPlus[i] = self.img.get(0, i) + self.column[i]
#pybind11#
#pybind11#        # get stats on the columns and rows
#pybind11#        self.imgProjectCol = afwMath.statisticsStack(self.img, afwMath.MEAN, 'x')
#pybind11#        self.imgProjectRow = afwMath.statisticsStack(self.img, afwMath.MEAN, 'y')
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.img
#pybind11#        del self.imgProjectCol
#pybind11#        del self.imgProjectRow
#pybind11#
#pybind11#    def testColumnStats(self):
#pybind11#        """Test the column statistics """
#pybind11#        for i in range(self.n):
#pybind11#            self.assertEqual(self.imgProjectCol.get(0, i)[0], self.column[i])
#pybind11#
#pybind11#    def testRowStats(self):
#pybind11#        """Test the row statistics """
#pybind11#        for i in range(self.n):
#pybind11#            self.assertEqual(self.imgProjectRow.get(i, 0)[0], self.row[i])
#pybind11#
#pybind11#    def testColumnOperators(self):
#pybind11#        """ Test operator overloading on columns """
#pybind11#
#pybind11#        columnSlice = afwImage.ImageSliceF(self.imgProjectCol.getImage())
#pybind11#
#pybind11#        imgAdd = self.img + columnSlice
#pybind11#        imgAdd2 = columnSlice + self.img
#pybind11#        imgSub = self.img - columnSlice
#pybind11#        imgMul = self.img * columnSlice
#pybind11#        imgMul2 = columnSlice * self.img
#pybind11#        imgDiv = self.img / columnSlice
#pybind11#
#pybind11#        for i in range(self.n):
#pybind11#            self.assertAlmostEqual(imgAdd.get(0, i), self.img.get(0, i) + columnSlice.get(0, i))
#pybind11#            self.assertAlmostEqual(imgAdd2.get(0, i), imgAdd.get(0, i))
#pybind11#            self.assertAlmostEqual(imgSub.get(0, i), self.img.get(0, i) - columnSlice.get(0, i))
#pybind11#            self.assertAlmostEqual(imgMul.get(0, i), self.img.get(0, i) * columnSlice.get(0, i))
#pybind11#            self.assertAlmostEqual(imgMul2.get(0, i), imgMul.get(0, i))
#pybind11#            self.assertAlmostEqual(imgDiv.get(0, i), self.img.get(0, i) / columnSlice.get(0, i))
#pybind11#
#pybind11#
#pybind11##################################################################
#pybind11## Test suite boiler plate
#pybind11##################################################################
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
