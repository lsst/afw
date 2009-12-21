#!/usr/bin/env python
"""
Tests for geom.BoxI, geom.BoxD

Run with:
   ./Coordinates.py
or
   python
   >>> import Coordinates; Coordinates.run()
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

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(BoxITestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
