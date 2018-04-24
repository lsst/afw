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
Tests for SpatialCell

Run with:
   python SpatialCell.py
or
   python
   >>> import SpatialCell; SpatialCell.run()
"""
import unittest

import lsst.utils.tests
import lsst.pex.exceptions as pexExcept
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom


def getFlux(x):
    return 1000 - 10*x


class SpatialCellTestCase(unittest.TestCase):
    """A test case for SpatialCell"""

    def setUp(self):
        candidateList = []
        self.nCandidate = 5
        for i in (0, 1, 4, 3, 2):       # must be all numbers in range(self.nCandidate)
            x, y = i, 5*i
            candidateList.append(afwMath.TestCandidate(x, y, getFlux(x)))

        self.cell = afwMath.SpatialCell("Test", afwGeom.Box2I(), candidateList)
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

        self.cell = afwMath.SpatialCell("Test", afwGeom.Box2I())

        for x, y in ([5, 0], [1, 1], [2, 2], [0, 0], [4, 4], [3, 4]):
            self.cell.insertCandidate(afwMath.TestCandidate(x, y, getFlux(x)))

        self.assertEqual(self.cell[0].getXCenter(), 0)

    def testIterators(self):
        """Test the SpatialCell iterators"""

        #
        # Count the candidates
        #
        self.assertEqual(self.cell.size(), self.nCandidate)
        self.assertEqual(self.cell.end() - self.cell.begin(), self.nCandidate)

        ptr = self.cell.begin()
        ptr.__incr__()
        self.assertEqual(self.cell.end() - ptr, self.nCandidate - 1)

        self.assertEqual(ptr - self.cell.begin(), 1)
        #
        # Now label one candidate as bad
        #
        self.cell[2].setStatus(afwMath.SpatialCellCandidate.BAD)

        self.assertEqual(self.cell.size(), self.nCandidate - 1)
        self.assertEqual(self.cell.end() - self.cell.begin(),
                         self.nCandidate - 1)

        self.cell.setIgnoreBad(False)
        self.assertEqual(self.cell.size(), self.nCandidate)
        self.assertEqual(self.cell.end() - self.cell.begin(), self.nCandidate)

    def testGetCandidateById(self):
        """Check that we can lookup candidates by ID"""
        id = self.cell[1].getId()
        self.assertEqual(self.cell.getCandidateById(id).getId(), id)

        self.assertEqual(self.cell.getCandidateById(-1, True), None)
        with self.assertRaises(pexExcept.NotFoundError):
            self.cell.getCandidateById(-1)

    def testSetIteratorBad(self):
        """Setting a candidate BAD shouldn't stop us seeing the rest of the candidates"""
        i = 0
        for cand in self.cell:
            if i == 1:
                cand.setStatus(afwMath.SpatialCellCandidate.BAD)
            i += 1

        self.assertEqual(i, self.nCandidate)

    def testSortCandidates(self):
        """Check that we can update ratings and maintain order"""
        ratings0 = [cand.getCandidateRating() for cand in self.cell]
        #
        # Change a rating
        #
        i, flux = 1, 9999
        self.cell[i].setCandidateRating(flux)
        ratings0[i] = flux

        self.assertEqual(ratings0, [cand.getCandidateRating()
                                    for cand in self.cell])

        self.cell.sortCandidates()
        self.assertNotEqual(
            ratings0, [cand.getCandidateRating() for cand in self.cell])

        def sortKey(a):
            return -a
        self.assertEqual(sorted(ratings0, key=sortKey),
                         [cand.getCandidateRating() for cand in self.cell])


class SpatialCellSetTestCase(unittest.TestCase):
    """A test case for SpatialCellSet"""

    def setUp(self):
        self.cellSet = afwMath.SpatialCellSet(afwGeom.Box2I(
            afwGeom.Point2I(0, 0), afwGeom.Extent2I(501, 501)), 260, 200)

    def makeTestCandidateCellSet(self):
        """Populate a SpatialCellSet"""

        if False:                       # Print the bboxes for the cells
            print()
            for i in range(len(self.cellSet.getCellList())):
                cell = self.cellSet.getCellList()[i]
                print(i, "%3d,%3d -- %3d,%3d" % (cell.getBBox().getMinX(), cell.getBBox().getMinY(),
                                                 cell.getBBox().getMaxX(), cell.getBBox().getMaxY()),
                      cell.getLabel())
        self.assertEqual(len(self.cellSet.getCellList()), 6)

        # number of candidates
        self.NTestCandidates = 0
        for x, y in ([5, 0], [1, 1], [2, 2], [0, 0], [4, 4], [3, 4]):  # all in cell0
            self.cellSet.insertCandidate(afwMath.TestCandidate(x, y, -x))
            self.NTestCandidates += 1

        # in cell1
        self.cellSet.insertCandidate(afwMath.TestCandidate(305, 0, 100))
        self.NTestCandidates += 1
        # the top right corner of cell5
        self.cellSet.insertCandidate(afwMath.TestCandidate(500, 500, 100))
        self.NTestCandidates += 1

    def tearDown(self):
        del self.cellSet

    def testNoCells(self):
        """Test that we check for a request to make a SpatialCellSet with no cells"""
        def tst():
            afwMath.SpatialCellSet(afwGeom.Box2I(
                afwGeom.Point2I(0, 0), afwGeom.Extent2I(500, 500)), 0, 3)

        self.assertRaises(pexExcept.LengthError, tst)

    def testInsertCandidate(self):
        """Insert candidates into the SpatialCellSet"""

        self.makeTestCandidateCellSet()

        def tst():
            # Doesn't fit
            self.cellSet.insertCandidate(afwMath.TestCandidate(501, 501, 100))
        self.assertRaises(pexExcept.OutOfRangeError, tst)
        #
        # OK, the SpatialCellList is populated
        #
        cell0 = self.cellSet.getCellList()[0]
        self.assertFalse(cell0.empty())
        self.assertEqual(cell0[0].getXCenter(), 0.0)

        self.assertEqual(self.cellSet.getCellList()[1][0].getXCenter(), 305.0)

        self.assertTrue(self.cellSet.getCellList()[2].empty())

        def tst1():
            self.cellSet.getCellList()[2][0]
        self.assertRaises(IndexError, tst1)

        def tst2():
            self.cellSet.getCellList()[2].begin().__deref__()
        self.assertRaises(pexExcept.NotFoundError, tst2)

        self.assertFalse(self.cellSet.getCellList()[5].empty())

    def testVisitor(self):
        """Test the candidate visitors"""

        self.makeTestCandidateCellSet()

        visitor = afwMath.TestCandidateVisitor()

        self.cellSet.visitCandidates(visitor)
        self.assertEqual(visitor.getN(), self.NTestCandidates)

        self.cellSet.visitCandidates(visitor, 1)
        self.assertEqual(visitor.getN(), 3)

    def testGetCandidateById(self):
        """Check that we can lookup candidates by ID"""

        self.makeTestCandidateCellSet()
        #
        # OK, the SpatialCellList is populated
        #
        id = self.cellSet.getCellList()[0][1].getId()
        self.assertEqual(self.cellSet.getCandidateById(id).getId(), id)

        def tst():
            self.cellSet.getCandidateById(-1)  # non-existent ID

        self.assertEqual(self.cellSet.getCandidateById(-1, True), None)
        self.assertRaises(pexExcept.NotFoundError, tst)

    def testSpatialCell(self):
        dx, dy, sx, sy = 100, 100, 50, 50
        for x0, y0 in [(0, 0), (100, 100)]:
            # only works for tests where dx,dx is some multiple of sx,sy
            assert(dx//sx == float(dx)/float(sx))
            assert(dy//sy == float(dy)/float(sy))

            bbox = afwGeom.Box2I(afwGeom.Point2I(x0, y0),
                                 afwGeom.Extent2I(dx, dy))
            cset = afwMath.SpatialCellSet(bbox, sx, sy)
            for cell in cset.getCellList():
                label = cell.getLabel()
                nx, ny = [int(z) for z in label.split()[1].split('x')]

                cbbox = cell.getBBox()

                self.assertEqual(cbbox.getMinX(), nx*sx + x0)
                self.assertEqual(cbbox.getMinY(), ny*sy + y0)
                self.assertEqual(cbbox.getMaxX(), (nx+1)*sx + x0 - 1)
                self.assertEqual(cbbox.getMaxY(), (ny+1)*sy + y0 - 1)

    def testSortCandidates(self):
        """Check that we can update ratings and maintain order"""

        self.makeTestCandidateCellSet()

        cell1 = self.cellSet.getCellList()[0]
        self.assertFalse(cell1.empty())

        ratings0 = [cand.getCandidateRating() for cand in cell1]
        #
        # Change a rating
        #
        i, flux = 1, 9999
        cell1[i].setCandidateRating(flux)
        ratings0[i] = flux

        self.assertEqual(
            ratings0, [cand.getCandidateRating() for cand in cell1])

        self.cellSet.sortCandidates()
        self.assertNotEqual(
            ratings0, [cand.getCandidateRating() for cand in cell1])

        def sortKey(a):
            return -a
        self.assertEqual(sorted(ratings0, key=sortKey),
                         [cand.getCandidateRating() for cand in cell1])


class TestImageCandidateCase(unittest.TestCase):
    """A test case for TestImageCandidate"""

    def setUp(self):
        self.cellSet = afwMath.SpatialCellSet(afwGeom.Box2I(
            afwGeom.Point2I(0, 0), afwGeom.Extent2I(501, 501)), 2, 3)

    def tearDown(self):
        del self.cellSet

    def testInsertCandidate(self):
        """Test that we can use SpatialCellMaskedImageCandidate"""

        flux = 10
        self.cellSet.insertCandidate(afwMath.TestImageCandidate(0, 0, flux))

        cand = self.cellSet.getCellList()[0][0]

        width, height = 15, 21
        cand.setWidth(width)
        cand.setHeight(height)

        im = cand.getMaskedImage().getImage()
        # This is how TestMaskedImageCandidate sets its pixels
        self.assertEqual(im.get(0, 0), flux)
        self.assertEqual(im.getWidth(), width)
        self.assertEqual(im.getHeight(), height)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
