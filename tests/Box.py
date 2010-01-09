#!/usr/bin/env python
"""
Tests for geom.BoxI, geom.BoxD

Run with:
   ./Box.py
or
   python
   >>> import Box; Box.run()
"""

import os
import pdb  # we may want to say pdb.set_trace()
import sys
import unittest
import numpy

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.afw.geom as geom

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class BoxITestCase(unittest.TestCase):

    def testEmpty(self):
        box = geom.BoxI()
        self.assert_(box.isEmpty())
        self.assertEqual(box.getWidth(), 0)
        self.assertEqual(box.getHeight(), 0)
        for x in (-1,0,1):
            for y in (-1,0,1):
                point = geom.makePointI(x,y)
                self.assertFalse(box.contains(point))
                box.include(point)
                self.assert_(box.contains(point))
                box = geom.BoxI()
        box.grow(3)
        self.assert_(box.isEmpty())

    def testConstruction(self):
        for n in range(10):
            xmin, xmax, ymin, ymax = numpy.random.randint(low=-5, high=5, size=4)
            if xmin > xmax: xmin, xmax = xmax, xmin
            if ymin > ymax: ymin, ymax = ymax, ymin
            pmin = geom.makePointI(xmin,ymin)
            pmax = geom.makePointI(xmax,ymax)
            # min/max constructor
            box = geom.BoxI(pmin,pmax)
            self.assertEqual(box.getMin(), pmin)
            self.assertEqual(box.getMax(), pmax)
            box = geom.BoxI(pmax,pmin)
            self.assertEqual(box.getMin(), pmin)
            self.assertEqual(box.getMax(), pmax)
            box = geom.BoxI(pmin,pmax,False)
            self.assertEqual(box.getMin(), pmin)
            self.assertEqual(box.getMax(), pmax)
            box = geom.BoxI(pmax,pmin,False)
            self.assert_(box.isEmpty() or pmax == pmin)
            # min/dim constructor
            dim = geom.ExtentI(1) + pmax - pmin
            if any(dim.eq(0)):
                box = geom.BoxI(pmin,dim)
                self.assert_(box.isEmpty())
                box = geom.BoxI(pmin,dim,False)
                self.assert_(box.isEmpty())
            else:
                box = geom.BoxI(pmin,dim)
                self.assertEqual(box.getMin(), pmin)
                self.assertEqual(box.getDimensions(), dim)
                box = geom.BoxI(pmin,dim,False)
                self.assertEqual(box.getMin(), pmin)
                self.assertEqual(box.getDimensions(), dim)
                dim = -dim
                box = geom.BoxI(pmin,dim)
                self.assertEqual(box.getMin(), pmin + dim + geom.ExtentI(1))
                self.assertEqual(box.getDimensions(),
                                 geom.makeExtentI(abs(dim.getX()),abs(dim.getY())))

    def testConversion(self):
        for n in range(10):
            xmin, xmax, ymin, ymax = numpy.random.uniform(low=-10, high=10, size=4)
            if xmin > xmax: xmin, xmax = xmax, xmin
            if ymin > ymax: ymin, ymax = ymax, ymin
            fpMin = geom.makePointD(xmin,ymin)
            fpMax = geom.makePointD(xmax,ymax)
            if any((fpMax-fpMin).lt(3)): continue  # avoid empty boxes
            fpBox = geom.BoxD(fpMin, fpMax)
            intBoxBig = geom.BoxI(fpBox, geom.BoxI.EXPAND)
            fpBoxBig = geom.BoxD(intBoxBig)
            intBoxSmall = geom.BoxI(fpBox, geom.BoxI.SHRINK)
            fpBoxSmall = geom.BoxD(intBoxSmall)
            self.assert_(fpBoxBig.contains(fpBox))
            self.assert_(fpBox.contains(fpBoxSmall))
            self.assert_(intBoxBig.contains(intBoxSmall))
            self.assert_(geom.BoxD(intBoxBig))
            self.assertEqual(geom.BoxI(fpBoxBig, geom.BoxI.EXPAND), intBoxBig)
            self.assertEqual(geom.BoxI(fpBoxSmall, geom.BoxI.SHRINK), intBoxSmall)

    def testAccessors(self):
        xmin, xmax, ymin, ymax = numpy.random.randint(low=-5, high=5, size=4)
        if xmin > xmax: xmin, xmax = xmax, xmin
        if ymin > ymax: ymin, ymax = ymax, ymin
        pmin = geom.makePointI(xmin,ymin)
        pmax = geom.makePointI(xmax,ymax)
        box = geom.BoxI(pmin, pmax, True)
        self.assertEqual(pmin, box.getMin())
        self.assertEqual(pmax, box.getMax())
        self.assertEqual(box.getMinX(), xmin)
        self.assertEqual(box.getMinY(), ymin)
        self.assertEqual(box.getMaxX(), xmax)
        self.assertEqual(box.getMaxY(), ymax)
        self.assertEqual(box.getBegin(), pmin)
        self.assertEqual(box.getEnd(), (pmax + geom.ExtentI(1)))
        self.assertEqual(box.getBeginX(), xmin)
        self.assertEqual(box.getBeginY(), ymin)
        self.assertEqual(box.getEndX(), xmax + 1)
        self.assertEqual(box.getEndY(), ymax + 1)
        self.assertEqual(box.getDimensions(), (pmax - pmin + geom.ExtentI(1)))
        self.assertEqual(box.getWidth(), (xmax - xmin  + 1))
        self.assertEqual(box.getHeight(), (ymax - ymin  + 1))
        self.assertEqual(box.getArea(), box.getWidth() * box.getHeight())
        
    def testRelations(self):
        box = geom.BoxI(geom.makePointI(-2,-3), geom.makePointI(2,1), True)
        self.assert_(box.contains(geom.makePointI(0,0)))
        self.assert_(box.contains(geom.makePointI(-2,-3)))
        self.assert_(box.contains(geom.makePointI(2,-3)))
        self.assert_(box.contains(geom.makePointI(2,1)))
        self.assert_(box.contains(geom.makePointI(-2,1)))
        self.assertFalse(box.contains(geom.makePointI(-2,-4)))
        self.assertFalse(box.contains(geom.makePointI(-3,-3)))
        self.assertFalse(box.contains(geom.makePointI(2,-4)))
        self.assertFalse(box.contains(geom.makePointI(3,-3)))
        self.assertFalse(box.contains(geom.makePointI(3,1)))
        self.assertFalse(box.contains(geom.makePointI(2,2)))
        self.assertFalse(box.contains(geom.makePointI(-3,1)))
        self.assertFalse(box.contains(geom.makePointI(-2,2)))
        self.assert_(box.contains(geom.BoxI(geom.makePointI(-1,-2), geom.makePointI(1,0))))
        self.assert_(box.contains(box))
        self.assertFalse(box.contains(geom.BoxI(geom.makePointI(-2,-3), geom.makePointI(2,2))))
        self.assertFalse(box.contains(geom.BoxI(geom.makePointI(-2,-3), geom.makePointI(3,1))))
        self.assertFalse(box.contains(geom.BoxI(geom.makePointI(-3,-3), geom.makePointI(2,1))))
        self.assertFalse(box.contains(geom.BoxI(geom.makePointI(-3,-4), geom.makePointI(2,1))))
        self.assert_(box.overlaps(geom.BoxI(geom.makePointI(-2,-3), geom.makePointI(2,2))))
        self.assert_(box.overlaps(geom.BoxI(geom.makePointI(-2,-3), geom.makePointI(3,1))))
        self.assert_(box.overlaps(geom.BoxI(geom.makePointI(-3,-3), geom.makePointI(2,1))))
        self.assert_(box.overlaps(geom.BoxI(geom.makePointI(-3,-4), geom.makePointI(2,1))))
        self.assert_(box.overlaps(geom.BoxI(geom.makePointI(-1,-2), geom.makePointI(1,0))))
        self.assert_(box.overlaps(box))
        self.assertFalse(box.overlaps(geom.BoxI(geom.makePointI(-5,-3), geom.makePointI(-3,1))))
        self.assertFalse(box.overlaps(geom.BoxI(geom.makePointI(-2,-6), geom.makePointI(2,-4))))
        self.assertFalse(box.overlaps(geom.BoxI(geom.makePointI(3,-3), geom.makePointI(4,1))))
        self.assertFalse(box.overlaps(geom.BoxI(geom.makePointI(-2,2), geom.makePointI(2,2))))

    def testMutators(self):
        box = geom.BoxI(geom.makePointI(-2,-3), geom.makePointI(2,1), True)
        box.grow(1)
        self.assertEqual(box, geom.BoxI(geom.makePointI(-3,-4), geom.makePointI(3,2), True))
        box.grow(geom.makeExtentI(2,3))
        self.assertEqual(box, geom.BoxI(geom.makePointI(-5,-7), geom.makePointI(5,5), True))
        box.shift(geom.makeExtentI(3,2))
        self.assertEqual(box, geom.BoxI(geom.makePointI(-2,-5), geom.makePointI(8,7), True))
        box.include(geom.makePointI(-4,2))
        self.assertEqual(box, geom.BoxI(geom.makePointI(-4,-5), geom.makePointI(8,7), True))
        box.include(geom.makePointI(0,-6))
        self.assertEqual(box, geom.BoxI(geom.makePointI(-4,-6), geom.makePointI(8,7), True))
        box.include(geom.BoxI(geom.makePointI(0,0), geom.makePointI(10,11), True))
        self.assertEqual(box, geom.BoxI(geom.makePointI(-4,-6), geom.makePointI(10,11), True))
        box.clip(geom.BoxI(geom.makePointI(0,0), geom.makePointI(11,12), True))
        self.assertEqual(box, geom.BoxI(geom.makePointI(0,0), geom.makePointI(10,11), True))
        box.clip(geom.BoxI(geom.makePointI(-1,-2), geom.makePointI(5,4), True))
        self.assertEqual(box, geom.BoxI(geom.makePointI(0,0), geom.makePointI(5,4), True))

class BoxDTestCase(unittest.TestCase):

    def testEmpty(self):
        box = geom.BoxD()
        self.assert_(box.isEmpty())
        self.assertEqual(box.getWidth(), 0.0)
        self.assertEqual(box.getHeight(), 0.0)
        for x in (-1,0,1):
            for y in (-1,0,1):
                point = geom.makePointD(x,y)
                self.assertFalse(box.contains(point))
                box.include(point)
                self.assert_(box.contains(point))
                box = geom.BoxD()
        box.grow(3)
        self.assert_(box.isEmpty())

    def testConstruction(self):
        for n in range(10):
            xmin, xmax, ymin, ymax = numpy.random.uniform(low=-5, high=5, size=4)
            if xmin > xmax: xmin, xmax = xmax, xmin
            if ymin > ymax: ymin, ymax = ymax, ymin
            pmin = geom.makePointD(xmin,ymin)
            pmax = geom.makePointD(xmax,ymax)
            # min/max constructor
            box = geom.BoxD(pmin,pmax)
            self.assertEqual(box.getMin(), pmin)
            self.assertEqual(box.getMax(), pmax)
            box = geom.BoxD(pmax,pmin)
            self.assertEqual(box.getMin(), pmin)
            self.assertEqual(box.getMax(), pmax)
            box = geom.BoxD(pmin,pmax,False)
            self.assertEqual(box.getMin(), pmin)
            self.assertEqual(box.getMax(), pmax)
            box = geom.BoxD(pmax,pmin,False)
            self.assert_(box.isEmpty())
            # min/dim constructor
            dim = pmax - pmin
            if any(dim.eq(0)):
                box = geom.BoxD(pmin,dim)
                self.assert_(box.isEmpty())
                box = geom.BoxD(pmin,dim,False)
                self.assert_(box.isEmpty())
            else:
                box = geom.BoxD(pmin,dim)
                self.assertEqual(box.getMin(), pmin)
                self.assertEqual(box.getDimensions(), dim)
                box = geom.BoxD(pmin,dim,False)
                self.assertEqual(box.getMin(), pmin)
                self.assertEqual(box.getDimensions(), dim)
                dim = -dim
                box = geom.BoxD(pmin,dim)
                self.assertEqual(box.getMin(), pmin + dim)
                self.assert_(numpy.allclose(box.getDimensions(),
                                            geom.makeExtentD(abs(dim.getX()),abs(dim.getY()))))

    def testAccessors(self):
        xmin, xmax, ymin, ymax = numpy.random.uniform(low=-5, high=5, size=4)
        if xmin > xmax: xmin, xmax = xmax, xmin
        if ymin > ymax: ymin, ymax = ymax, ymin
        pmin = geom.makePointD(xmin,ymin)
        pmax = geom.makePointD(xmax,ymax)
        box = geom.BoxD(pmin, pmax, True)
        self.assertEqual(pmin, box.getMin())
        self.assertEqual(pmax, box.getMax())
        self.assertEqual(box.getMinX(), xmin)
        self.assertEqual(box.getMinY(), ymin)
        self.assertEqual(box.getMaxX(), xmax)
        self.assertEqual(box.getMaxY(), ymax)
        self.assertEqual(box.getDimensions(), (pmax - pmin))
        self.assertEqual(box.getWidth(), (xmax - xmin))
        self.assertEqual(box.getHeight(), (ymax - ymin))
        self.assertEqual(box.getArea(), box.getWidth() * box.getHeight())
        self.assertEqual(box.getCenterX(), 0.5*(pmax.getX() + pmin.getX()))
        self.assertEqual(box.getCenterY(), 0.5*(pmax.getY() + pmin.getY()))
        self.assertEqual(box.getCenter().getX(), box.getCenterX())
        self.assertEqual(box.getCenter().getY(), box.getCenterY())
        
    def testRelations(self):
        box = geom.BoxD(geom.makePointD(-2,-3), geom.makePointD(2,1), True)
        self.assert_(box.contains(geom.makePointD(0,0)))
        self.assert_(box.contains(geom.makePointD(-2,-3)))
        self.assertFalse(box.contains(geom.makePointD(2,-3)))
        self.assertFalse(box.contains(geom.makePointD(2,1)))
        self.assertFalse(box.contains(geom.makePointD(-2,1)))
        self.assert_(box.contains(geom.BoxD(geom.makePointD(-1,-2), geom.makePointD(1,0))))
        self.assert_(box.contains(box))
        self.assertFalse(box.contains(geom.BoxD(geom.makePointD(-2,-3), geom.makePointD(2,2))))
        self.assertFalse(box.contains(geom.BoxD(geom.makePointD(-2,-3), geom.makePointD(3,1))))
        self.assertFalse(box.contains(geom.BoxD(geom.makePointD(-3,-3), geom.makePointD(2,1))))
        self.assertFalse(box.contains(geom.BoxD(geom.makePointD(-3,-4), geom.makePointD(2,1))))
        self.assert_(box.overlaps(geom.BoxD(geom.makePointD(-2,-3), geom.makePointD(2,2))))
        self.assert_(box.overlaps(geom.BoxD(geom.makePointD(-2,-3), geom.makePointD(3,1))))
        self.assert_(box.overlaps(geom.BoxD(geom.makePointD(-3,-3), geom.makePointD(2,1))))
        self.assert_(box.overlaps(geom.BoxD(geom.makePointD(-3,-4), geom.makePointD(2,1))))
        self.assert_(box.overlaps(geom.BoxD(geom.makePointD(-1,-2), geom.makePointD(1,0))))
        self.assert_(box.overlaps(box))
        self.assertFalse(box.overlaps(geom.BoxD(geom.makePointD(-5,-3), geom.makePointD(-3,1))))
        self.assertFalse(box.overlaps(geom.BoxD(geom.makePointD(-2,-6), geom.makePointD(2,-4))))
        self.assertFalse(box.overlaps(geom.BoxD(geom.makePointD(3,-3), geom.makePointD(4,1))))
        self.assertFalse(box.overlaps(geom.BoxD(geom.makePointD(-2,2), geom.makePointD(2,2))))
        self.assertFalse(box.overlaps(geom.BoxD(geom.makePointD(-2,-5), geom.makePointD(2,-3))))
        self.assertFalse(box.overlaps(geom.BoxD(geom.makePointD(-4,-3), geom.makePointD(-2,1))))
        self.assertFalse(box.contains(geom.BoxD(geom.makePointD(-2,1), geom.makePointD(2,3))))
        self.assertFalse(box.contains(geom.BoxD(geom.makePointD(2,-3), geom.makePointD(4,1))))

    def testMutators(self):
        box = geom.BoxD(geom.makePointD(-2,-3), geom.makePointD(2,1), True)
        box.grow(1)
        self.assertEqual(box, geom.BoxD(geom.makePointD(-3,-4), geom.makePointD(3,2), True))
        box.grow(geom.makeExtentD(2,3))
        self.assertEqual(box, geom.BoxD(geom.makePointD(-5,-7), geom.makePointD(5,5), True))
        box.shift(geom.makeExtentD(3,2))
        self.assertEqual(box, geom.BoxD(geom.makePointD(-2,-5), geom.makePointD(8,7), True))
        box.include(geom.makePointD(-4,2))
        self.assertEqual(box, geom.BoxD(geom.makePointD(-4,-5), geom.makePointD(8,7), True))
        self.assert_(box.contains(geom.makePointD(-4,2)))
        box.include(geom.makePointD(0,-6))
        self.assertEqual(box, geom.BoxD(geom.makePointD(-4,-6), geom.makePointD(8,7), True))
        box.include(geom.BoxD(geom.makePointD(0,0), geom.makePointD(10,11), True))
        self.assertEqual(box, geom.BoxD(geom.makePointD(-4,-6), geom.makePointD(10,11), True))
        box.clip(geom.BoxD(geom.makePointD(0,0), geom.makePointD(11,12), True))
        self.assertEqual(box, geom.BoxD(geom.makePointD(0,0), geom.makePointD(10,11), True))
        box.clip(geom.BoxD(geom.makePointD(-1,-2), geom.makePointD(5,4), True))
        self.assertEqual(box, geom.BoxD(geom.makePointD(0,0), geom.makePointD(5,4), True))


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(BoxITestCase)
    suites += unittest.makeSuite(BoxDTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
