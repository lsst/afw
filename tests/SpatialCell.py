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
        self.assertEqual(self.cell[0].getXCenter(), 0)
        self.assertEqual(self.cell[1].getXCenter(), 1)
        self.assertEqual(self.cell[1].getYCenter(), 5)

    def testBuildCandidateListByInsertion(self):
        """Build a candidate list by inserting candidates"""

        self.cell = afwMath.SpatialCell("Test", afwImage.BBox())
        for x, y in ([5, 0], [1, 1], [2, 2], [0, 0], [4, 4], [3, 4]):
            self.cell.insertCandidate(testLib.TestCandidate(x, y, getFlux(x)))

        self.assertEqual(self.cell[0].getXCenter(), 0)

    def testIterators(self):
        """Test the SpatialCell iterators"""
        #
        # Count the candidates
        #
        self.assertEqual(self.cell.size(), self.nCandidate)
        self.assertEqual(self.cell.end() - self.cell.begin(), self.nCandidate)

        ptr = self.cell.begin(); ptr.__incr__()
        self.assertEqual(self.cell.end() - ptr, self.nCandidate - 1)
        
        self.assertEqual(ptr - self.cell.begin(), 1)
        #
        # Now label one candidate as bad
        #
        ptr = self.cell[2].setStatus(afwMath.SpatialCellCandidate.BAD)

        self.assertEqual(self.cell.size(), self.nCandidate - 1)
        self.assertEqual(self.cell.end() - self.cell.begin(), self.nCandidate - 1)

        self.cell.setIgnoreBad(False)
        self.assertEqual(self.cell.size(), self.nCandidate)
        self.assertEqual(self.cell.end() - self.cell.begin(), self.nCandidate)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SpatialCellSetTestCase(unittest.TestCase):
    """A test case for SpatialCellSet"""

    def setUp(self):
        self.cellSet = afwMath.SpatialCellSet(afwImage.BBox(afwImage.PointI(0, 0), 501, 501), 260, 200)

    def tearDown(self):
        del self.cellSet

    def testNoCells(self):
        """Test that we check for a request to make a SpatialCellSet with no cells"""
        def tst():
            afwMath.SpatialCellSet(afwImage.BBox(afwImage.PointI(0, 0), 500, 500), 0, 3)

        utilsTests.assertRaisesLsstCpp(self, pexExcept.LengthErrorException, tst)

    def testInsertCandidate(self):
        """Insert candidates into the SpatialCellSet"""

        if False:                       # Print the bboxes for the cells
            print
            for i in range(len(self.cellSet.getCellList())):
                cell = self.cellSet.getCellList()[i]
                print i, "%3d,%3d -- %3d,%3d" % (cell.getBBox().getX0(), cell.getBBox().getY0(),
                                                 cell.getBBox().getX1(), cell.getBBox().getY1()), cell
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
        cell0 = self.cellSet.getCellList()[0]
        self.assertFalse(cell0.empty())
        self.assertEqual(cell0[0].getXCenter(), 0.0)
        
        self.assertEqual(self.cellSet.getCellList()[1][0].getXCenter(), 305.0)

        self.assertTrue(self.cellSet.getCellList()[2].empty())

        def tst():
            self.cellSet.getCellList()[2][0]
        self.assertRaises(IndexError, tst)

        def tst():
            self.cellSet.getCellList()[2].begin().__deref__()
        utilsTests.assertRaisesLsstCpp(self, pexExcept.NotFoundException, tst)

        self.assertFalse(self.cellSet.getCellList()[5].empty())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class TestImageCandidateCase(unittest.TestCase):
    """A test case for TestImageCandidate"""

    def setUp(self):
        self.cellSet = afwMath.SpatialCellSet(afwImage.BBox(afwImage.PointI(0, 0), 501, 501), 2, 3)

    def tearDown(self):
        del self.cellSet

    def testInsertCandidate(self):
        """Test that we can use SpatialCellImageCandidate"""

        flux = 10
        self.cellSet.insertCandidate(testLib.TestImageCandidate(0, 0, flux))

        cand = self.cellSet.getCellList()[0][0]
        #
        # Swig doesn't know that we're a SpatialCellImageCandidate;  all it knows is that we have
        # a SpatialCellCandidate, and SpatialCellCandidates don't know about getImage;  so cast the
        # pointer to SpatialCellImageCandidate<Image<float> > and all will be well;
        #
        # First check that we _can't_ cast to SpatialCellImageCandidate<MaskedImage<float> >
        #
        self.assertEqual(afwMath.cast_SpatialCellImageCandidateMF(cand), None)

        cand = afwMath.cast_SpatialCellImageCandidateF(cand)

        width, height = 15, 21
        cand.setWidth(width); cand.setHeight(height);

        im = cand.getImage()
        self.assertEqual(im.get(0,0), flux) # This is how TestImageCandidate sets its pixels
        self.assertEqual(im.getWidth(), width)
        self.assertEqual(im.getHeight(), height)
        
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(SpatialCellTestCase)
    suites += unittest.makeSuite(SpatialCellSetTestCase)
    suites += unittest.makeSuite(TestImageCandidateCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
