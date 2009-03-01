#!/usr/bin/env python
"""
Tests for SpatialCell

Run with:
   python SpatialCell.py
or
   python
   >>> import SpatialCell; SpatialCell.run()
"""

import os
import pdb  # we may want to say pdb.set_trace()
import sys
import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept
import lsst.afw.image.imageLib as afwImage
import lsst.afw.math.mathLib as afwMath
import lsst.afw.display.ds9 as ds9

import testLib

try:
    type(display)
except NameError:
    display = False

def getFlux(x):
    return 1000 - 10*x

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SpatialCellTestCase(unittest.TestCase):
    """A test case for SpatialCell"""

    def setUp(self):
        candidateList = afwMath.SpatialCellCandidateList()
        self.nCandidate = 5
        for i in (0, 1, 4, 3, 2):       # must be all numbers in range(self.nCandidate)
            x, y = i, 5*i
            candidateList.append(testLib.TestCandidate(x, y, getFlux(x)))
    
        self.cell = afwMath.SpatialCell("Test", afwImage.BBox(), candidateList)
        self.assertEqual(self.cell.getLabel(), "Test")

    def tearDown(self):
        del self.cell

    def testCandidateList(self):
        """Check that we can retrieve candidates, and that they are sorted by ranking"""
        self.assertEqual(self.cell.getCurrentCandidate().getXCenter(), 0)

        self.cell.nextCandidate()
        self.assertEqual(self.cell.getCurrentCandidate().getXCenter(), 1)
        self.assertEqual(self.cell.getCurrentCandidate().getYCenter(), 5)

    def testCandidateList2(self):
        """Check that we can traverse the candidate list, but not fall off the end"""

        self.assertEqual(self.cell.isUsable(), True)
        for i in range(1, self.nCandidate):
            self.assertEqual(self.cell.nextCandidate(), True)
            self.assertEqual(self.cell.isUsable(), True)

        for i in range(2):              # or any range for that matter
            self.assertEqual(self.cell.nextCandidate(), False)
            self.assertEqual(self.cell.isUsable(), False)
        #
        # Let's go back one
        #
        self.cell.prevCandidate()
        self.assertEqual(self.cell.isUsable(), True)
        self.assertEqual(self.cell.getCurrentCandidate().getXCenter(), 4)
        #
        # And back to the beginning
        #
        self.cell.prevCandidate(True)
        self.assertEqual(self.cell.getCurrentCandidate().getXCenter(), 0)

    def testBuildCandidateListByInsertion(self):
        """Build a candidate list by inserting candidates"""

        self.cell = afwMath.SpatialCell("Test", afwImage.BBox())
        for x, y in ([5, 0], [1, 1], [2, 2], [0, 0], [4, 4], [3, 4]):
            self.cell.insertCandidate(testLib.TestCandidate(x, y, getFlux(x)))
        #
        # The first candidate installed will be current
        #
        self.assertEqual(self.cell.getCurrentCandidate().getXCenter(), 5)

        self.cell.prevCandidate(True)
        self.assertEqual(self.cell.getCurrentCandidate().getXCenter(), 0)

    def testInsertCandidateList(self):
        """Check that inserting new candidates doesn't invalidate currentCandidate, and
        preserves sort order"""

        self.cell.nextCandidate()
        self.cell.nextCandidate()
        self.assertEqual(self.cell.getCurrentCandidate().getXCenter(), 2)

        x, y = 2.5, 0
        self.cell.insertCandidate(testLib.TestCandidate(x, y, getFlux(x)))
        self.assertEqual(self.cell.getCurrentCandidate().getXCenter(), 2)

        self.cell.nextCandidate()
        self.assertEqual(self.cell.getCurrentCandidate().getXCenter(), 2.5)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SpatialCellSetTestCase(unittest.TestCase):
    """A test case for SpatialCellSet"""

    def setUp(self):
        self.cellSet = afwMath.SpatialCellSet(afwImage.BBox(afwImage.PointI(0, 0), 501, 501), 2, 3)

    def tearDown(self):
        del self.cellSet

    def testNoCells(self):
        """Test that we check for a request to make a SpatialCellSet with no cells"""
        def tst():
            afwMath.SpatialCellSet(afwImage.BBox(afwImage.PointI(0, 0), 500, 500), 0, 3)

        utilsTests.assertRaisesLsstCpp(self, pexExcept.LengthErrorException, tst)

    def testInsertCandidate(self):
        """Insert candidates into the SpatialCellSet"""

        self.assertEqual(len(self.cellSet.getCellList()), 6)

        for x, y in ([5, 0], [1, 1], [2, 2], [0, 0], [4, 4], [3, 4]): # all in cell0
            self.cellSet.insertCandidate(testLib.TestCandidate(x, y, -x))

        self.cellSet.insertCandidate(testLib.TestCandidate(305, 0, 100))        # in cell1
        self.cellSet.insertCandidate(testLib.TestCandidate(500, 500, 100))      # the top right corner of cell5

        def tst():
            self.cellSet.insertCandidate(testLib.TestCandidate(501, 501, 100))      # Doesn't fit
        utilsTests.assertRaisesLsstCpp(self, pexExcept.OutOfRangeException, tst)
        #
        # OK, the SpatialCellList is populated
        #
        if False:
            print
            for i in range(len(self.cellSet.getCellList())):
                cell = self.cellSet.getCellList()[i]
                print i, "%3d,%3d -- %3d,%3d" % (cell.getBBox().getX0(), cell.getBBox().getY0(),
                                                 cell.getBBox().getX1(), cell.getBBox().getY1()), cell.isUsable()

        self.assertEqual(self.cellSet.getCellList()[0].isUsable(), True)
        self.assertEqual(self.cellSet.getCellList()[0].getCurrentCandidate().getXCenter(), 5.0)
        
        self.assertEqual(self.cellSet.getCellList()[1].getCurrentCandidate().getXCenter(), 305.0)

        def tst():
            self.cellSet.getCellList()[2].getCurrentCandidate()

        self.assertEqual(self.cellSet.getCellList()[2].isUsable(), False)
        utilsTests.assertRaisesLsstCpp(self, pexExcept.NotFoundException, tst)

        self.assertEqual(self.cellSet.getCellList()[5].isUsable(), True)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(SpatialCellTestCase)
    suites += unittest.makeSuite(SpatialCellSetTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
