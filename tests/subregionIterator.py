#!/usr/bin/env python

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

import unittest

import numpy

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept
import lsst.pex.logging as pexLog
import lsst.afw.geom as afwGeom
import lsst.afw.math.detail as mathDetail
import lsst.afw.image.testUtils as imTestUtils

VERBOSITY = 0 # increase to see trace

pexLog.Debug("lsst.afw", VERBOSITY)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SubregionIteratorTestCase(unittest.TestCase):
    def setUp(self):
        self.region = afwGeom.BoxI(afwGeom.makePointI(4, 5), afwGeom.makeExtentI(100, 102))
        self.regionStart = self.region.getMin()
        self.regionEnd = self.region.getMax()
        self.overlap = afwGeom.makeExtentI(3, 4)
        self.subregionSize = afwGeom.makeExtentI(20, 21)
        self.regionIter = mathDetail.SubregionIterator(self.region, self.subregionSize, self.overlap)

    def tearDown(self):
        self.region = None
        self.regionStart = None
        self.regionEnd = None
        self.overlap = None
        self.subregionSize = None
        self.regionIter = None

    def testIllegalCases(self):
        bboxSize = afwGeom.makeExtentI(10, 11)
        for bboxStart in (
            afwGeom.makePointI(self.regionStart.getX() - 1, self.regionStart.getY()),
            afwGeom.makePointI(self.regionStart.getX() - 2, self.regionStart.getY()),
            afwGeom.makePointI(self.regionStart.getX(), self.regionStart.getY() - 1),
            afwGeom.makePointI(self.regionStart.getX(), self.regionStart.getY() - 2),
        ):
            bbox = afwGeom.BoxI(bboxStart, bboxSize)
            self.assertRaises(pexExcept.LsstCppException, self.regionIter.getNext, bbox)

        for bboxEnd in (
            afwGeom.makePointI(self.regionEnd.getX() + 1, self.regionEnd.getY()),
            afwGeom.makePointI(self.regionEnd.getX() + 2, self.regionEnd.getY()),
            afwGeom.makePointI(self.regionEnd.getX(), self.regionEnd.getY() + 1),
            afwGeom.makePointI(self.regionEnd.getX(), self.regionEnd.getY() + 2),
        ):
            bbox = afwGeom.BoxI(self.regionEnd, bboxEnd)
            self.assertRaises(pexExcept.LsstCppException, self.regionIter.getNext, bbox)

    def testIsEnd(self):
        for bboxStart in (
            afwGeom.makePointI(self.regionEnd.getX(), self.regionEnd.getY()),
            afwGeom.makePointI(self.regionEnd.getX() - 10, self.regionEnd.getY()),
            afwGeom.makePointI(self.regionEnd.getX(), self.regionEnd.getY() - 10),
            afwGeom.makePointI(self.regionStart.getX(), self.regionStart.getY()),
        ):
            bbox = afwGeom.BoxI(bboxStart, self.regionEnd)
            nextBBox = self.regionIter.getNext(bbox)
            self.assertTrue(self.regionIter.isEnd(nextBBox))

        for bboxStart in (
            self.regionStart,
            self.regionEnd - afwGeom.makeExtentI(10, 10),
        ):
            for bboxEnd in (
                afwGeom.makePointI(self.regionEnd.getX() - 1, self.regionEnd.getY()),
                afwGeom.makePointI(self.regionEnd.getX() - 2, self.regionEnd.getY()),
                afwGeom.makePointI(self.regionEnd.getX(), self.regionEnd.getY() - 1),
                afwGeom.makePointI(self.regionEnd.getX(), self.regionEnd.getY() - 2),
                afwGeom.makePointI(self.regionEnd.getX() - 1, self.regionEnd.getY() - 1),
                afwGeom.makePointI(self.regionEnd.getX() - 2, self.regionEnd.getY() - 2),
            ):
                bbox = afwGeom.BoxI(bboxStart, bboxEnd)
                nextBBox = self.regionIter.getNext(bbox)
                self.assertFalse(self.regionIter.isEnd(nextBBox))

    def testGetNext(self):
        for bboxEnd in (
            afwGeom.makePointI(self.regionEnd.getX() - 1, self.regionEnd.getY()),
            afwGeom.makePointI(self.regionEnd.getX() - 2, self.regionEnd.getY()),
            afwGeom.makePointI(self.regionEnd.getX(), self.regionEnd.getY() - 1),
            afwGeom.makePointI(self.regionEnd.getX(), self.regionEnd.getY() - 2),
            afwGeom.makePointI(self.regionEnd.getX() - 1, self.regionEnd.getY() - 1),
            afwGeom.makePointI(self.regionEnd.getX() - 2, self.regionEnd.getY() - 2),
        ):
            bbox = afwGeom.BoxI(self.regionStart, bboxEnd)
            nextBBox = self.regionIter.getNext(bbox)

            # basic tests
            self.assertTrue(nextBBox.getWidth() > self.overlap.getX())
            self.assertTrue(nextBBox.getHeight() > self.overlap.getY())
            self.assertFalse(self.regionIter.isEnd(nextBBox))

            # verify starting x,y of nextBBox
            if bbox.getMaxX() == self.regionEnd.getX():
                # start new row
                self.assertTrue(nextBBox.getMinX() == self.regionStart.getX())
                self.assertTrue(nextBBox.getMinY() == bbox.getMaxY() + 1 - self.overlap.getY())
            else:
                # continue on same row
                self.assertTrue(nextBBox.getMinX() == bbox.getMaxX() + 1 - self.overlap.getX())
                self.assertTrue(nextBBox.getMinY() == bbox.getMinY())

            # verify size or ending x,y of nextBBox
            if nextBBox.getMaxX() < self.region.getMaxX():
                self.assertTrue(nextBBox.getWidth() == self.subregionSize.getX())
            else:
                self.assertTrue(nextBBox.getMaxX() == self.region.getMaxX())
            if nextBBox.getMaxY() < self.region.getMaxY():
                self.assertTrue(nextBBox.getHeight() == self.subregionSize.getY())
            else:
                self.assertTrue(nextBBox.getMaxY() == self.region.getMaxY())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(SubregionIteratorTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(doExit=False):
    """Run the tests"""
    utilsTests.run(suite(), doExit)

if __name__ == "__main__":
    run(True)
