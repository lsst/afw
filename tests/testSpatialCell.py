#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import range
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
#pybind11#"""
#pybind11#Tests for SpatialCell
#pybind11#
#pybind11#Run with:
#pybind11#   python SpatialCell.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import SpatialCell; SpatialCell.run()
#pybind11#"""
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#
#pybind11#import testLib
#pybind11#
#pybind11#
#pybind11#def getFlux(x):
#pybind11#    return 1000 - 10*x
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class SpatialCellTestCase(unittest.TestCase):
#pybind11#    """A test case for SpatialCell"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        candidateList = afwMath.SpatialCellCandidateList()
#pybind11#        self.nCandidate = 5
#pybind11#        for i in (0, 1, 4, 3, 2):       # must be all numbers in range(self.nCandidate)
#pybind11#            x, y = i, 5*i
#pybind11#            candidateList.append(testLib.TestCandidate(x, y, getFlux(x)))
#pybind11#
#pybind11#        self.cell = afwMath.SpatialCell("Test", afwGeom.Box2I(), candidateList)
#pybind11#        self.assertEqual(self.cell.getLabel(), "Test")
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.cell
#pybind11#
#pybind11#    def testCandidateList(self):
#pybind11#        """Check that we can retrieve candidates, and that they are sorted by ranking"""
#pybind11#        self.assertEqual(self.cell[0].getXCenter(), 0)
#pybind11#        self.assertEqual(self.cell[1].getXCenter(), 1)
#pybind11#        self.assertEqual(self.cell[1].getYCenter(), 5)
#pybind11#
#pybind11#    def testBuildCandidateListByInsertion(self):
#pybind11#        """Build a candidate list by inserting candidates"""
#pybind11#
#pybind11#        self.cell = afwMath.SpatialCell("Test", afwGeom.Box2I())
#pybind11#
#pybind11#        for x, y in ([5, 0], [1, 1], [2, 2], [0, 0], [4, 4], [3, 4]):
#pybind11#            self.cell.insertCandidate(testLib.TestCandidate(x, y, getFlux(x)))
#pybind11#
#pybind11#        self.assertEqual(self.cell[0].getXCenter(), 0)
#pybind11#
#pybind11#    def testIterators(self):
#pybind11#        """Test the SpatialCell iterators"""
#pybind11#
#pybind11#        #
#pybind11#        # Count the candidates
#pybind11#        #
#pybind11#        self.assertEqual(self.cell.size(), self.nCandidate)
#pybind11#        self.assertEqual(self.cell.end() - self.cell.begin(), self.nCandidate)
#pybind11#
#pybind11#        ptr = self.cell.begin()
#pybind11#        ptr.__incr__()
#pybind11#        self.assertEqual(self.cell.end() - ptr, self.nCandidate - 1)
#pybind11#
#pybind11#        self.assertEqual(ptr - self.cell.begin(), 1)
#pybind11#        #
#pybind11#        # Now label one candidate as bad
#pybind11#        #
#pybind11#        self.cell[2].setStatus(afwMath.SpatialCellCandidate.BAD)
#pybind11#
#pybind11#        self.assertEqual(self.cell.size(), self.nCandidate - 1)
#pybind11#        self.assertEqual(self.cell.end() - self.cell.begin(), self.nCandidate - 1)
#pybind11#
#pybind11#        self.cell.setIgnoreBad(False)
#pybind11#        self.assertEqual(self.cell.size(), self.nCandidate)
#pybind11#        self.assertEqual(self.cell.end() - self.cell.begin(), self.nCandidate)
#pybind11#
#pybind11#    def testGetCandidateById(self):
#pybind11#        """Check that we can lookup candidates by ID"""
#pybind11#        id = self.cell[1].getId()
#pybind11#        self.assertEqual(self.cell.getCandidateById(id).getId(), id)
#pybind11#
#pybind11#        self.assertEqual(self.cell.getCandidateById(-1, True), None)
#pybind11#        with self.assertRaises(pexExcept.NotFoundError):
#pybind11#            self.cell.getCandidateById(-1)
#pybind11#
#pybind11#    def testSetIteratorBad(self):
#pybind11#        """Setting a candidate BAD shouldn't stop us seeing the rest of the candidates"""
#pybind11#        i = 0
#pybind11#        for cand in self.cell:
#pybind11#            if i == 1:
#pybind11#                cand.setStatus(afwMath.SpatialCellCandidate.BAD)
#pybind11#            i += 1
#pybind11#
#pybind11#        self.assertEqual(i, self.nCandidate)
#pybind11#
#pybind11#    def testSortCandidates(self):
#pybind11#        """Check that we can update ratings and maintain order"""
#pybind11#        ratings0 = [cand.getCandidateRating() for cand in self.cell]
#pybind11#        #
#pybind11#        # Change a rating
#pybind11#        #
#pybind11#        i, flux = 1, 9999
#pybind11#        self.cell[i].setCandidateRating(flux)
#pybind11#        ratings0[i] = flux
#pybind11#
#pybind11#        self.assertEqual(ratings0, [cand.getCandidateRating() for cand in self.cell])
#pybind11#
#pybind11#        self.cell.sortCandidates()
#pybind11#        self.assertNotEqual(ratings0, [cand.getCandidateRating() for cand in self.cell])
#pybind11#        def sortKey(a):
#pybind11#            return -a
#pybind11#        self.assertEqual(sorted(ratings0, key=sortKey),
#pybind11#                         [cand.getCandidateRating() for cand in self.cell])
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class SpatialCellSetTestCase(unittest.TestCase):
#pybind11#    """A test case for SpatialCellSet"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.cellSet = afwMath.SpatialCellSet(afwGeom.Box2I(
#pybind11#            afwGeom.Point2I(0, 0), afwGeom.Extent2I(501, 501)), 260, 200)
#pybind11#
#pybind11#    def makeTestCandidateCellSet(self):
#pybind11#        """Populate a SpatialCellSet"""
#pybind11#
#pybind11#        if False:                       # Print the bboxes for the cells
#pybind11#            print()
#pybind11#            for i in range(len(self.cellSet.getCellList())):
#pybind11#                cell = self.cellSet.getCellList()[i]
#pybind11#                print(i, "%3d,%3d -- %3d,%3d" % (cell.getBBox().getMinX(), cell.getBBox().getMinY(),
#pybind11#                                                 cell.getBBox().getMaxX(), cell.getBBox().getMaxY()),
#pybind11#                      cell.getLabel())
#pybind11#        self.assertEqual(len(self.cellSet.getCellList()), 6)
#pybind11#
#pybind11#        self.NTestCandidates = 0                                      # number of candidates
#pybind11#        for x, y in ([5, 0], [1, 1], [2, 2], [0, 0], [4, 4], [3, 4]):  # all in cell0
#pybind11#            self.cellSet.insertCandidate(testLib.TestCandidate(x, y, -x))
#pybind11#            self.NTestCandidates += 1
#pybind11#
#pybind11#        self.cellSet.insertCandidate(testLib.TestCandidate(305, 0, 100))   # in cell1
#pybind11#        self.NTestCandidates += 1
#pybind11#        self.cellSet.insertCandidate(testLib.TestCandidate(500, 500, 100))  # the top right corner of cell5
#pybind11#        self.NTestCandidates += 1
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.cellSet
#pybind11#
#pybind11#    def testNoCells(self):
#pybind11#        """Test that we check for a request to make a SpatialCellSet with no cells"""
#pybind11#        def tst():
#pybind11#            afwMath.SpatialCellSet(afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(500, 500)), 0, 3)
#pybind11#
#pybind11#        self.assertRaises(pexExcept.LengthError, tst)
#pybind11#
#pybind11#    def testInsertCandidate(self):
#pybind11#        """Insert candidates into the SpatialCellSet"""
#pybind11#
#pybind11#        self.makeTestCandidateCellSet()
#pybind11#
#pybind11#        def tst():
#pybind11#            self.cellSet.insertCandidate(testLib.TestCandidate(501, 501, 100))      # Doesn't fit
#pybind11#        self.assertRaises(pexExcept.OutOfRangeError, tst)
#pybind11#        #
#pybind11#        # OK, the SpatialCellList is populated
#pybind11#        #
#pybind11#        cell0 = self.cellSet.getCellList()[0]
#pybind11#        self.assertFalse(cell0.empty())
#pybind11#        self.assertEqual(cell0[0].getXCenter(), 0.0)
#pybind11#
#pybind11#        self.assertEqual(self.cellSet.getCellList()[1][0].getXCenter(), 305.0)
#pybind11#
#pybind11#        self.assertTrue(self.cellSet.getCellList()[2].empty())
#pybind11#
#pybind11#        def tst1():
#pybind11#            self.cellSet.getCellList()[2][0]
#pybind11#        self.assertRaises(IndexError, tst1)
#pybind11#
#pybind11#        def tst2():
#pybind11#            self.cellSet.getCellList()[2].begin().__deref__()
#pybind11#        self.assertRaises(pexExcept.NotFoundError, tst2)
#pybind11#
#pybind11#        self.assertFalse(self.cellSet.getCellList()[5].empty())
#pybind11#
#pybind11#    def testVisitor(self):
#pybind11#        """Test the candidate visitors"""
#pybind11#
#pybind11#        self.makeTestCandidateCellSet()
#pybind11#
#pybind11#        visitor = testLib.TestCandidateVisitor()
#pybind11#
#pybind11#        self.cellSet.visitCandidates(visitor)
#pybind11#        self.assertEqual(visitor.getN(), self.NTestCandidates)
#pybind11#
#pybind11#        self.cellSet.visitCandidates(visitor, 1)
#pybind11#        self.assertEqual(visitor.getN(), 3)
#pybind11#
#pybind11#    def testGetCandidateById(self):
#pybind11#        """Check that we can lookup candidates by ID"""
#pybind11#
#pybind11#        self.makeTestCandidateCellSet()
#pybind11#        #
#pybind11#        # OK, the SpatialCellList is populated
#pybind11#        #
#pybind11#        id = self.cellSet.getCellList()[0][1].getId()
#pybind11#        self.assertEqual(self.cellSet.getCandidateById(id).getId(), id)
#pybind11#
#pybind11#        def tst():
#pybind11#            self.cellSet.getCandidateById(-1)  # non-existent ID
#pybind11#
#pybind11#        self.assertEqual(self.cellSet.getCandidateById(-1, True), None)
#pybind11#        self.assertRaises(pexExcept.NotFoundError, tst)
#pybind11#
#pybind11#    def testSpatialCell(self):
#pybind11#        dx, dy, sx, sy = 100, 100, 50, 50
#pybind11#        for x0, y0 in [(0, 0), (100, 100)]:
#pybind11#            # only works for tests where dx,dx is some multiple of sx,sy
#pybind11#            assert(dx//sx == float(dx)/float(sx))
#pybind11#            assert(dy//sy == float(dy)/float(sy))
#pybind11#
#pybind11#            bbox = afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Extent2I(dx, dy))
#pybind11#            cset = afwMath.SpatialCellSet(bbox, sx, sy)
#pybind11#            for cell in cset.getCellList():
#pybind11#                label = cell.getLabel()
#pybind11#                nx, ny = [int(z) for z in label.split()[1].split('x')]
#pybind11#
#pybind11#                cbbox = cell.getBBox()
#pybind11#
#pybind11#                self.assertEqual(cbbox.getMinX(), nx*sx + x0)
#pybind11#                self.assertEqual(cbbox.getMinY(), ny*sy + y0)
#pybind11#                self.assertEqual(cbbox.getMaxX(), (nx+1)*sx + x0 - 1)
#pybind11#                self.assertEqual(cbbox.getMaxY(), (ny+1)*sy + y0 - 1)
#pybind11#
#pybind11#    def testSortCandidates(self):
#pybind11#        """Check that we can update ratings and maintain order"""
#pybind11#
#pybind11#        self.makeTestCandidateCellSet()
#pybind11#
#pybind11#        cell1 = self.cellSet.getCellList()[0]
#pybind11#        self.assertFalse(cell1.empty())
#pybind11#
#pybind11#        ratings0 = [cand.getCandidateRating() for cand in cell1]
#pybind11#        #
#pybind11#        # Change a rating
#pybind11#        #
#pybind11#        i, flux = 1, 9999
#pybind11#        cell1[i].setCandidateRating(flux)
#pybind11#        ratings0[i] = flux
#pybind11#
#pybind11#        self.assertEqual(ratings0, [cand.getCandidateRating() for cand in cell1])
#pybind11#
#pybind11#        self.cellSet.sortCandidates()
#pybind11#        self.assertNotEqual(ratings0, [cand.getCandidateRating() for cand in cell1])
#pybind11#        def sortKey(a):
#pybind11#            return -a
#pybind11#        self.assertEqual(sorted(ratings0, key=sortKey),
#pybind11#                         [cand.getCandidateRating() for cand in cell1])
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class TestMaskedImageCandidateCase(unittest.TestCase):
#pybind11#    """A test case for TestMaskedImageCandidate"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.cellSet = afwMath.SpatialCellSet(afwGeom.Box2I(
#pybind11#            afwGeom.Point2I(0, 0), afwGeom.Extent2I(501, 501)), 2, 3)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.cellSet
#pybind11#
#pybind11#    def testInsertCandidate(self):
#pybind11#        """Test that we can use SpatialCellMaskedImageCandidate"""
#pybind11#
#pybind11#        flux = 10
#pybind11#        self.cellSet.insertCandidate(testLib.TestMaskedImageCandidate(0, 0, flux))
#pybind11#
#pybind11#        cand = self.cellSet.getCellList()[0][0]
#pybind11#        #
#pybind11#        # Swig doesn't know that we're a SpatialCellMaskedImageCandidate;  all it knows is that we have
#pybind11#        # a SpatialCellCandidate, and SpatialCellCandidates don't know about getMaskedImage;  so cast the
#pybind11#        # pointer to SpatialCellMaskedImageCandidate<Image<float> > and all will be well;
#pybind11#        #
#pybind11#
#pybind11#        cand = afwMath.SpatialCellMaskedImageCandidateF.cast(cand)
#pybind11#
#pybind11#        width, height = 15, 21
#pybind11#        cand.setWidth(width)
#pybind11#        cand.setHeight(height)
#pybind11#
#pybind11#        im = cand.getMaskedImage().getImage()
#pybind11#        self.assertEqual(im.get(0, 0), flux)  # This is how TestMaskedImageCandidate sets its pixels
#pybind11#        self.assertEqual(im.getWidth(), width)
#pybind11#        self.assertEqual(im.getHeight(), height)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
