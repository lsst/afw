#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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

"""
Tests for Statistics

Run with:
   ./statisticsOverloads.py
or
   python
   >>> import statisticsOverloads; statisticsOverloads.run()
"""

import unittest

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.geom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath

try:
    type(display)
except NameError:
    display = False


class StatisticsTestCase(unittest.TestCase):

    """A test case to check all overloaded makeStatistics() factories for Statistics"""

    def setUp(self):
        self.val = 10
        self.nRow, self.nCol = 100, 200
        self.sctrl = afwMath.StatisticsControl()

        # Integers
        self.mimgI = afwImage.MaskedImageI(
            lsst.geom.Extent2I(self.nRow, self.nCol))
        self.mimgI.set(self.val, 0x0, self.val)
        self.imgI = afwImage.ImageI(
            lsst.geom.Extent2I(self.nRow, self.nCol), self.val)
        # TODO: pybind11, this should probably be ndarray
        self.vecI = [self.val for i in range(self.nRow*self.nCol)]

        # floats
        self.mimgF = afwImage.MaskedImageF(
            lsst.geom.Extent2I(self.nRow, self.nCol))
        self.mimgF.set(self.val, 0x0, self.val)
        self.imgF = afwImage.ImageF(
            lsst.geom.Extent2I(self.nRow, self.nCol), self.val)
        # TODO: pybind11, this should probably be ndarray
        self.vecF = [float(self.val) for i in range(self.nRow*self.nCol)]

        # doubles
        self.mimgD = afwImage.MaskedImageD(
            lsst.geom.Extent2I(self.nRow, self.nCol))
        self.mimgD.set(self.val, 0x0, self.val)
        self.imgD = afwImage.ImageD(
            lsst.geom.Extent2I(self.nRow, self.nCol), self.val)
        # TODO: pybind11, this should probably be ndarray
        self.vecD = [float(self.val) for i in range(self.nRow*self.nCol)]

        self.imgList = [self.imgI, self.imgF, self.imgD]
        self.mimgList = [self.mimgI, self.mimgF, self.mimgD]
        self.vecList = [self.vecI, self.vecF, self.vecD]

    def tearDown(self):
        del self.mimgI
        del self.mimgF
        del self.mimgD
        del self.imgI
        del self.imgF
        del self.imgD
        del self.vecI
        del self.vecF
        del self.vecD

        del self.mimgList
        del self.imgList
        del self.vecList

    # The guts of the testing: grab a mean, stddev, and sum for whatever
    # you're called with
    def compareMakeStatistics(self, image, n):
        stats = afwMath.makeStatistics(image, afwMath.NPOINT | afwMath.STDEV |
                                       afwMath.MEAN | afwMath.SUM, self.sctrl)

        self.assertEqual(stats.getValue(afwMath.NPOINT), n)
        self.assertEqual(stats.getValue(afwMath.NPOINT)*stats.getValue(afwMath.MEAN),
                         stats.getValue(afwMath.SUM))
        self.assertEqual(stats.getValue(afwMath.MEAN), self.val)
        self.assertEqual(stats.getValue(afwMath.STDEV), 0)

    # same as compareMakeStatistics but calls constructor directly (only for
    # masked image)
    def compareStatistics(self, stats, n):
        self.assertEqual(stats.getValue(afwMath.NPOINT), n)
        self.assertEqual(stats.getValue(afwMath.NPOINT)*stats.getValue(afwMath.MEAN),
                         stats.getValue(afwMath.SUM))
        self.assertEqual(stats.getValue(afwMath.MEAN), self.val)
        self.assertEqual(stats.getValue(afwMath.STDEV), 0)

    # Test regular image::Image
    def testImage(self):
        for img in self.imgList:
            self.compareMakeStatistics(img, img.getWidth()*img.getHeight())

    # Test the image::MaskedImages
    def testMaskedImage(self):
        for mimg in self.mimgList:
            self.compareMakeStatistics(mimg, mimg.getWidth()*mimg.getHeight())

    # Test the std::vectors
    def testVector(self):
        for vec in self.vecList:
            self.compareMakeStatistics(vec, len(vec))

    def testWeightedVector(self):
        """Test std::vector, but with weights"""
        sctrl = afwMath.StatisticsControl()

        nval = len(self.vecList[0])
        weight = 10
        weights = [i*weight/float(nval - 1) for i in range(nval)]

        for vec in self.vecList:
            stats = afwMath.makeStatistics(vec, weights,
                                           afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM, sctrl)

            self.assertAlmostEqual(
                0.5*weight*sum(vec)/stats.getValue(afwMath.SUM), 1.0)
            self.assertAlmostEqual(
                sum(vec)/len(vec), stats.getValue(afwMath.MEAN))

    # Try calling the Statistics constructor directly
    def testStatisticsConstructor(self):
        if False:
            statsI = afwMath.StatisticsI(self.mimgI.getImage(), self.mimgI.getMask(),
                                         afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM,
                                         self.sctrl)
            statsF = afwMath.StatisticsF(self.mimgF.getImage(), self.mimgF.getMask(),
                                         afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM,
                                         self.sctrl)
            statsD = afwMath.StatisticsD(self.mimgD.getImage(), self.mimgD.getMask(),
                                         afwMath.NPOINT | afwMath.STDEV | afwMath.MEAN | afwMath.SUM,
                                         self.sctrl)

            self.compareStatistics(
                statsI, self.mimgI.getWidth()*self.mimgI.getHeight())
            self.compareStatistics(
                statsF, self.mimgF.getWidth()*self.mimgF.getHeight())
            self.compareStatistics(
                statsD, self.mimgD.getWidth()*self.mimgD.getHeight())

    # Test the Mask specialization
    def testMask(self):
        mask = afwImage.Mask(lsst.geom.Extent2I(10, 10))
        mask.set(0x0)

        mask[1, 1, afwImage.LOCAL] = 0x10
        mask[3, 1, afwImage.LOCAL] = 0x08
        mask[5, 4, afwImage.LOCAL] = 0x08
        mask[4, 5, afwImage.LOCAL] = 0x02

        stats = afwMath.makeStatistics(mask, afwMath.SUM | afwMath.NPOINT)
        self.assertEqual(mask.getWidth()*mask.getHeight(),
                         stats.getValue(afwMath.NPOINT))
        self.assertEqual(0x1a, stats.getValue(afwMath.SUM))

        def tst():
            afwMath.makeStatistics(mask, afwMath.MEAN)
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, tst)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
