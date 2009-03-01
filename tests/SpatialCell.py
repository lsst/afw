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

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SpatialCellTestCase(unittest.TestCase):
    """A test case for SpatialCell"""

    def getFlux(self, x):
        return 1000 - 10*x

    def setUp(self):
        candidateList = afwMath.SpatialCellCandidateList()
        self.nCandidate = 5
        for i in (0, 1, 4, 3, 2):       # must be all numbers in range(self.nCandidate)
            x, y = i, 5*i
            candidateList.append(testLib.TestCandidate(x, y, self.getFlux(x)))
    
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
            self.cell.insertCandidate(testLib.TestCandidate(x, y, self.getFlux(x)))
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
        self.cell.insertCandidate(testLib.TestCandidate(x, y, self.getFlux(x)))
        self.assertEqual(self.cell.getCurrentCandidate().getXCenter(), 2)

        self.cell.nextCandidate()
        self.assertEqual(self.cell.getCurrentCandidate().getXCenter(), 2.5)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(SpatialCellTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
