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
                point = geom.PointI.makeXY(x,y)
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
            pmin = geom.PointI.makeXY(xmin,ymin)
            pmax = geom.PointI.makeXY(xmax,ymax)
            # min/max constructor
            box = geom.BoxI(pmin,pmax)
            self.assert_(all(box.getMin() == pmin))
            self.assert_(all(box.getMax() == pmax))
            box = geom.BoxI(pmax,pmin)
            self.assert_(all(box.getMin() == pmin))
            self.assert_(all(box.getMax() == pmax))
            box = geom.BoxI(pmin,pmax,False)
            self.assert_(all(box.getMin() == pmin))
            self.assert_(all(box.getMax() == pmax))
            box = geom.BoxI(pmax,pmin,False)
            self.assert_(box.isEmpty())
            # min/dim constructor
            dim = geom.ExtentI(1) + pmax - pmin
            if any(dim == 0):
                box = geom.BoxI(pmin,dim)
                self.assert_(box.isEmpty())
                box = geom.BoxI(pmin,dim,False)
                self.assert_(box.isEmpty())
            else:
                box = geom.BoxI(pmin,dim)
                self.assert_(all(box.getMin() == pmin))
                self.assert_(all(box.getDimensions() == dim))
                box = geom.BoxI(pmin,dim,False)
                self.assert_(all(box.getMin() == pmin))
                self.assert_(all(box.getDimensions() == dim))
                dim = -dim
                box = geom.BoxI(pmin,dim)
                self.assert_(all(box.getMin() == pmin + dim + geom.ExtentI(1)))
                self.assert_(all(box.getDimensions() 
                                 == geom.ExtentI.makeXY(abs(dim.getX()),abs(dim.getY()))))

    def testConversion(self):
        for n in range(10):
            xmin, xmax, ymin, ymax = numpy.random.uniform(low=-10, high=10, size=4)
            if xmin > xmax: xmin, xmax = xmax, xmin
            if ymin > ymax: ymin, ymax = ymax, ymin
            fpMin = geom.PointD.makeXY(xmin,ymin)
            fpMax = geom.PointD.makeXY(xmax,ymax)
            if any(fpMax-fpMin < 3): continue  # avoid empty boxes
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
        pmin = geom.PointI.makeXY(xmin,ymin)
        pmax = geom.PointI.makeXY(xmax,ymax)
        box = geom.BoxI(pmin, pmax, True)
        self.assert_(all(pmin == box.getMin()))
        self.assert_(all(pmax == box.getMax()))
        self.assertEqual(box.getMinX(), xmin)
        self.assertEqual(box.getMinY(), ymin)
        self.assertEqual(box.getMaxX(), xmax)
        self.assertEqual(box.getMaxY(), ymax)
        self.assert_(all(box.getBegin() == pmin))
        self.assert_(all(box.getEnd() == (pmax + geom.ExtentI(1))))
        self.assertEqual(box.getBeginX(), xmin)
        self.assertEqual(box.getBeginY(), ymin)
        self.assertEqual(box.getEndX(), xmax + 1)
        self.assertEqual(box.getEndY(), ymax + 1)
        self.assert_(all(box.getDimensions() == (pmax - pmin + geom.ExtentI(1))))
        self.assertEqual(box.getWidth(), (xmax - xmin  + 1))
        self.assertEqual(box.getHeight(), (ymax - ymin  + 1))
        self.assertEqual(box.getArea(), box.getWidth() * box.getHeight())
        
    def testRelations(self):
        box = geom.BoxI(geom.PointI.makeXY(-2,-3), geom.PointI.makeXY(2,1), True)
        self.assert_(box.contains(geom.PointI.makeXY(0,0)))
        self.assert_(box.contains(geom.PointI.makeXY(-2,-3)))
        self.assert_(box.contains(geom.PointI.makeXY(2,-3)))
        self.assert_(box.contains(geom.PointI.makeXY(2,1)))
        self.assert_(box.contains(geom.PointI.makeXY(-2,1)))
        self.assertFalse(box.contains(geom.PointI.makeXY(-2,-4)))
        self.assertFalse(box.contains(geom.PointI.makeXY(-3,-3)))
        self.assertFalse(box.contains(geom.PointI.makeXY(2,-4)))
        self.assertFalse(box.contains(geom.PointI.makeXY(3,-3)))
        self.assertFalse(box.contains(geom.PointI.makeXY(3,1)))
        self.assertFalse(box.contains(geom.PointI.makeXY(2,2)))
        self.assertFalse(box.contains(geom.PointI.makeXY(-3,1)))
        self.assertFalse(box.contains(geom.PointI.makeXY(-2,2)))
        self.assert_(box.contains(geom.BoxI(geom.PointI.makeXY(-1,-2), geom.PointI.makeXY(1,0))))
        self.assert_(box.contains(box))
        self.assertFalse(box.contains(geom.BoxI(geom.PointI.makeXY(-2,-3), geom.PointI.makeXY(2,2))))
        self.assertFalse(box.contains(geom.BoxI(geom.PointI.makeXY(-2,-3), geom.PointI.makeXY(3,1))))
        self.assertFalse(box.contains(geom.BoxI(geom.PointI.makeXY(-3,-3), geom.PointI.makeXY(2,1))))
        self.assertFalse(box.contains(geom.BoxI(geom.PointI.makeXY(-3,-4), geom.PointI.makeXY(2,1))))
        self.assert_(box.overlaps(geom.BoxI(geom.PointI.makeXY(-2,-3), geom.PointI.makeXY(2,2))))
        self.assert_(box.overlaps(geom.BoxI(geom.PointI.makeXY(-2,-3), geom.PointI.makeXY(3,1))))
        self.assert_(box.overlaps(geom.BoxI(geom.PointI.makeXY(-3,-3), geom.PointI.makeXY(2,1))))
        self.assert_(box.overlaps(geom.BoxI(geom.PointI.makeXY(-3,-4), geom.PointI.makeXY(2,1))))
        self.assert_(box.overlaps(geom.BoxI(geom.PointI.makeXY(-1,-2), geom.PointI.makeXY(1,0))))
        self.assert_(box.overlaps(box))
        self.assertFalse(box.overlaps(geom.BoxI(geom.PointI.makeXY(-5,-3), geom.PointI.makeXY(-3,1))))
        self.assertFalse(box.overlaps(geom.BoxI(geom.PointI.makeXY(-2,-6), geom.PointI.makeXY(2,-4))))
        self.assertFalse(box.overlaps(geom.BoxI(geom.PointI.makeXY(3,-3), geom.PointI.makeXY(4,1))))
        self.assertFalse(box.overlaps(geom.BoxI(geom.PointI.makeXY(-2,2), geom.PointI.makeXY(2,2))))

    def testMutators(self):
        box = geom.BoxI(geom.PointI.makeXY(-2,-3), geom.PointI.makeXY(2,1), True)
        box.grow(1)
        self.assertEqual(box, geom.BoxI(geom.PointI.makeXY(-3,-4), geom.PointI.makeXY(3,2), True))
        box.grow(geom.ExtentI.makeXY(2,3))
        self.assertEqual(box, geom.BoxI(geom.PointI.makeXY(-5,-7), geom.PointI.makeXY(5,5), True))
        box.shift(geom.ExtentI.makeXY(3,2))
        self.assertEqual(box, geom.BoxI(geom.PointI.makeXY(-2,-5), geom.PointI.makeXY(8,7), True))
        box.include(geom.PointI.makeXY(-4,2))
        self.assertEqual(box, geom.BoxI(geom.PointI.makeXY(-4,-5), geom.PointI.makeXY(8,7), True))
        box.include(geom.PointI.makeXY(0,-6))
        self.assertEqual(box, geom.BoxI(geom.PointI.makeXY(-4,-6), geom.PointI.makeXY(8,7), True))
        box.include(geom.BoxI(geom.PointI.makeXY(0,0), geom.PointI.makeXY(10,11), True))
        self.assertEqual(box, geom.BoxI(geom.PointI.makeXY(-4,-6), geom.PointI.makeXY(10,11), True))
        box.clip(geom.BoxI(geom.PointI.makeXY(0,0), geom.PointI.makeXY(11,12), True))
        self.assertEqual(box, geom.BoxI(geom.PointI.makeXY(0,0), geom.PointI.makeXY(10,11), True))
        box.clip(geom.BoxI(geom.PointI.makeXY(-1,-2), geom.PointI.makeXY(5,4), True))
        self.assertEqual(box, geom.BoxI(geom.PointI.makeXY(0,0), geom.PointI.makeXY(5,4), True))


class BoxDTestCase(unittest.TestCase):

    def testEmpty(self):
        box = geom.BoxD()
        self.assert_(box.isEmpty())
        self.assertEqual(box.getWidth(), 0.0)
        self.assertEqual(box.getHeight(), 0.0)
        for x in (-1,0,1):
            for y in (-1,0,1):
                point = geom.PointD.makeXY(x,y)
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
            pmin = geom.PointD.makeXY(xmin,ymin)
            pmax = geom.PointD.makeXY(xmax,ymax)
            # min/max constructor
            box = geom.BoxD(pmin,pmax)
            self.assert_(all(box.getMin() == pmin))
            self.assert_(all(box.getMax() == pmax))
            box = geom.BoxD(pmax,pmin)
            self.assert_(all(box.getMin() == pmin))
            self.assert_(all(box.getMax() == pmax))
            box = geom.BoxD(pmin,pmax,False)
            self.assert_(all(box.getMin() == pmin))
            self.assert_(all(box.getMax() == pmax))
            box = geom.BoxD(pmax,pmin,False)
            self.assert_(box.isEmpty())
            # min/dim constructor
            dim = pmax - pmin
            if any(dim == 0):
                box = geom.BoxD(pmin,dim)
                self.assert_(box.isEmpty())
                box = geom.BoxD(pmin,dim,False)
                self.assert_(box.isEmpty())
            else:
                box = geom.BoxD(pmin,dim)
                self.assert_(all(box.getMin() == pmin))
                self.assert_(all(box.getDimensions() == dim))
                box = geom.BoxD(pmin,dim,False)
                self.assert_(all(box.getMin() == pmin))
                self.assert_(all(box.getDimensions() == dim))
                dim = -dim
                box = geom.BoxD(pmin,dim)
                self.assert_(all(box.getMin() == pmin + dim))
                self.assert_(numpy.allclose(box.getDimensions(),
                                            geom.ExtentD.makeXY(abs(dim.getX()),abs(dim.getY()))))

    def testAccessors(self):
        xmin, xmax, ymin, ymax = numpy.random.uniform(low=-5, high=5, size=4)
        if xmin > xmax: xmin, xmax = xmax, xmin
        if ymin > ymax: ymin, ymax = ymax, ymin
        pmin = geom.PointD.makeXY(xmin,ymin)
        pmax = geom.PointD.makeXY(xmax,ymax)
        box = geom.BoxD(pmin, pmax, True)
        self.assert_(all(pmin == box.getMin()))
        self.assert_(all(pmax == box.getMax()))
        self.assertEqual(box.getMinX(), xmin)
        self.assertEqual(box.getMinY(), ymin)
        self.assertEqual(box.getMaxX(), xmax)
        self.assertEqual(box.getMaxY(), ymax)
        self.assert_(all(box.getDimensions() == (pmax - pmin)))
        self.assertEqual(box.getWidth(), (xmax - xmin))
        self.assertEqual(box.getHeight(), (ymax - ymin))
        self.assertEqual(box.getArea(), box.getWidth() * box.getHeight())
        self.assertEqual(box.getCenterX(), 0.5*(pmax.getX() + pmin.getX()))
        self.assertEqual(box.getCenterY(), 0.5*(pmax.getY() + pmin.getY()))
        self.assertEqual(box.getCenter().getX(), box.getCenterX())
        self.assertEqual(box.getCenter().getY(), box.getCenterY())
        
    def testRelations(self):
        box = geom.BoxD(geom.PointD.makeXY(-2,-3), geom.PointD.makeXY(2,1), True)
        self.assert_(box.contains(geom.PointD.makeXY(0,0)))
        self.assert_(box.contains(geom.PointD.makeXY(-2,-3)))
        self.assertFalse(box.contains(geom.PointD.makeXY(2,-3)))
        self.assertFalse(box.contains(geom.PointD.makeXY(2,1)))
        self.assertFalse(box.contains(geom.PointD.makeXY(-2,1)))
        self.assert_(box.contains(geom.BoxD(geom.PointD.makeXY(-1,-2), geom.PointD.makeXY(1,0))))
        self.assert_(box.contains(box))
        self.assertFalse(box.contains(geom.BoxD(geom.PointD.makeXY(-2,-3), geom.PointD.makeXY(2,2))))
        self.assertFalse(box.contains(geom.BoxD(geom.PointD.makeXY(-2,-3), geom.PointD.makeXY(3,1))))
        self.assertFalse(box.contains(geom.BoxD(geom.PointD.makeXY(-3,-3), geom.PointD.makeXY(2,1))))
        self.assertFalse(box.contains(geom.BoxD(geom.PointD.makeXY(-3,-4), geom.PointD.makeXY(2,1))))
        self.assert_(box.overlaps(geom.BoxD(geom.PointD.makeXY(-2,-3), geom.PointD.makeXY(2,2))))
        self.assert_(box.overlaps(geom.BoxD(geom.PointD.makeXY(-2,-3), geom.PointD.makeXY(3,1))))
        self.assert_(box.overlaps(geom.BoxD(geom.PointD.makeXY(-3,-3), geom.PointD.makeXY(2,1))))
        self.assert_(box.overlaps(geom.BoxD(geom.PointD.makeXY(-3,-4), geom.PointD.makeXY(2,1))))
        self.assert_(box.overlaps(geom.BoxD(geom.PointD.makeXY(-1,-2), geom.PointD.makeXY(1,0))))
        self.assert_(box.overlaps(box))
        self.assertFalse(box.overlaps(geom.BoxD(geom.PointD.makeXY(-5,-3), geom.PointD.makeXY(-3,1))))
        self.assertFalse(box.overlaps(geom.BoxD(geom.PointD.makeXY(-2,-6), geom.PointD.makeXY(2,-4))))
        self.assertFalse(box.overlaps(geom.BoxD(geom.PointD.makeXY(3,-3), geom.PointD.makeXY(4,1))))
        self.assertFalse(box.overlaps(geom.BoxD(geom.PointD.makeXY(-2,2), geom.PointD.makeXY(2,2))))
        self.assertFalse(box.overlaps(geom.BoxD(geom.PointD.makeXY(-2,-5), geom.PointD.makeXY(2,-3))))
        self.assertFalse(box.overlaps(geom.BoxD(geom.PointD.makeXY(-4,-3), geom.PointD.makeXY(-2,1))))
        self.assertFalse(box.contains(geom.BoxD(geom.PointD.makeXY(-2,1), geom.PointD.makeXY(2,3))))
        self.assertFalse(box.contains(geom.BoxD(geom.PointD.makeXY(2,-3), geom.PointD.makeXY(4,1))))

    def testMutators(self):
        box = geom.BoxD(geom.PointD.makeXY(-2,-3), geom.PointD.makeXY(2,1), True)
        box.grow(1)
        self.assertEqual(box, geom.BoxD(geom.PointD.makeXY(-3,-4), geom.PointD.makeXY(3,2), True))
        box.grow(geom.ExtentD.makeXY(2,3))
        self.assertEqual(box, geom.BoxD(geom.PointD.makeXY(-5,-7), geom.PointD.makeXY(5,5), True))
        box.shift(geom.ExtentD.makeXY(3,2))
        self.assertEqual(box, geom.BoxD(geom.PointD.makeXY(-2,-5), geom.PointD.makeXY(8,7), True))
        box.include(geom.PointD.makeXY(-4,2))
        self.assertEqual(box, geom.BoxD(geom.PointD.makeXY(-4,-5), geom.PointD.makeXY(8,7), True))
        self.assert_(box.contains(geom.PointD.makeXY(-4,2)))
        box.include(geom.PointD.makeXY(0,-6))
        self.assertEqual(box, geom.BoxD(geom.PointD.makeXY(-4,-6), geom.PointD.makeXY(8,7), True))
        box.include(geom.BoxD(geom.PointD.makeXY(0,0), geom.PointD.makeXY(10,11), True))
        self.assertEqual(box, geom.BoxD(geom.PointD.makeXY(-4,-6), geom.PointD.makeXY(10,11), True))
        box.clip(geom.BoxD(geom.PointD.makeXY(0,0), geom.PointD.makeXY(11,12), True))
        self.assertEqual(box, geom.BoxD(geom.PointD.makeXY(0,0), geom.PointD.makeXY(10,11), True))
        box.clip(geom.BoxD(geom.PointD.makeXY(-1,-2), geom.PointD.makeXY(5,4), True))
        self.assertEqual(box, geom.BoxD(geom.PointD.makeXY(0,0), geom.PointD.makeXY(5,4), True))


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
