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

"""
Tests for HeavyFootprints

Run with:
   heavyFootprint.py
or
   python
   >>> import heavyFootprint; heavyFootprint.run()
"""

import numpy as np
import math, sys
import unittest
import lsst.utils.tests as tests
import lsst.pex.logging as logging
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDetect
import lsst.afw.geom as afwGeom
import lsst.afw.display.ds9 as ds9

try:
    type(verbose)
except NameError:
    verbose = 0
    logging.Debug("afwDetect.Footprint", verbose)

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        
class HeavyFootprintTestCase(unittest.TestCase):
    """A test case for HeavyFootprint"""
    def setUp(self):
        self.mi = afwImage.MaskedImageF(20, 10)
        self.objectPixelVal = (10, 0x1, 100)
        
        self.foot = afwDetect.Footprint()
        for y, x0, x1 in [(2, 10, 13),
                          (3, 11, 14)]:
            self.foot.addSpan(y, x0, x1)

            for x in range(x0, x1 + 1):
                self.mi.set(x, y, self.objectPixelVal)

    def tearDown(self):
        del self.foot
        del self.mi

    def testCreate(self):
        """Check that we can create a HeavyFootprint"""

        imi = self.mi.Factory(self.mi, True) # copy of input image

        hfoot = afwDetect.makeHeavyFootprint(self.foot, self.mi)
        self.assertNotEqual(hfoot.getId(), None) # check we can call a base-class method
        #
        # Check we didn't modify the input image
        #
        self.assertTrue(np.all(np.equal(self.mi.getImage().getArray(), imi.getImage().getArray())))
        
        omi = self.mi.Factory(self.mi.getDimensions())
        omi.set((1, 0x4, 0.1))
        hfoot.insert(omi)

        if display:
            ds9.mtv(imi, frame=0, title="input")
            ds9.mtv(omi, frame=1, title="output")

        for s in self.foot.getSpans():
            y = s.getY()
            for x in range(s.getX0(), s.getX1() + 1):
                self.assertEqual(imi.get(x, y), omi.get(x, y))

        # Check that we can call getImageArray(), etc
        arr = hfoot.getImageArray()
        print arr
        # Check that it's iterable
        for x in arr:
            pass
        arr = hfoot.getMaskArray()
        print arr
        for x in arr:
            pass
        arr = hfoot.getVarianceArray()
        print arr
        # Check that it's iterable
        for x in arr:
            pass
        


    def testSetFootprint(self):
        """Check that we can create a HeavyFootprint and set the pixels under it"""

        ctrl = afwDetect.HeavyFootprintCtrl()
        ctrl.setModifySource(afwDetect.HeavyFootprintCtrl.SET) # clear the pixels in the Footprint
        ctrl.setMaskVal(self.objectPixelVal[1])

        hfoot = afwDetect.makeHeavyFootprint(self.foot, self.mi, ctrl)
        #
        # Check that we cleared all the pixels
        #
        self.assertEqual(np.min(self.mi.getImage().getArray()), 0.0)
        self.assertEqual(np.max(self.mi.getImage().getArray()), 0.0)
        self.assertEqual(np.min(self.mi.getMask().getArray()), 0.0)
        self.assertEqual(np.max(self.mi.getMask().getArray()), 0.0)
        self.assertEqual(np.min(self.mi.getVariance().getArray()), 0.0)
        self.assertEqual(np.max(self.mi.getVariance().getArray()), 0.0)

    def testMakeHeavy(self):
        """Test that we can make a FootprintSet heavy"""
        fs = afwDetect.FootprintSet(self.mi, afwDetect.Threshold(1))

        ctrl = afwDetect.HeavyFootprintCtrl(afwDetect.HeavyFootprintCtrl.NONE)
        fs.makeHeavy(self.mi, ctrl)

        if display:
            ds9.mtv(self.mi, frame=0, title="input")
            #ds9.mtv(omi, frame=1, title="output")

        omi = self.mi.Factory(self.mi.getDimensions())

        for foot in fs.getFootprints():
            self.assertNotEqual(afwDetect.cast_HeavyFootprint(foot, self.mi), None)
            afwDetect.cast_HeavyFootprint(foot, self.mi).insert(omi)

        for foot in fs.getFootprints():
            self.assertNotEqual(afwDetect.HeavyFootprintF.cast(foot), None)
            afwDetect.HeavyFootprintF.cast(foot).insert(omi)

        self.assertTrue(np.all(np.equal(self.mi.getImage().getArray(), omi.getImage().getArray())))

    def testMakeHeavy(self):
        """Test that inserting a HeavyFootprint obeys XY0"""
        fs = afwDetect.FootprintSet(self.mi, afwDetect.Threshold(1))

        fs.makeHeavy(self.mi)

        bbox = afwGeom.BoxI(afwGeom.PointI(9, 1), afwGeom.ExtentI(7, 4))
        omi = self.mi.Factory(self.mi, bbox, afwImage.LOCAL, True)
        omi.set((0, 0x0, 0))

        for foot in fs.getFootprints():
            afwDetect.cast_HeavyFootprint(foot, self.mi).insert(omi)

        if display:
            ds9.mtv(self.mi, frame=0, title="input")
            ds9.mtv(omi, frame=1, title="sub")

        submi = self.mi.Factory(self.mi, bbox)
        self.assertTrue(np.all(np.equal(submi.getImage().getArray(), omi.getImage().getArray())))

    def testCast_HeavyFootprint(self):
        """Test that we can cast a Footprint to a HeavyFootprint"""

        hfoot = afwDetect.makeHeavyFootprint(self.foot, self.mi)

        ctrl = afwDetect.HeavyFootprintCtrl(afwDetect.HeavyFootprintCtrl.NONE)
        hfoot = afwDetect.makeHeavyFootprint(self.foot, self.mi, ctrl)
        #
        # This isn't quite a full test, as hfoot is already a HeavyFootprint,
        # the complete test is in testMakeHeavy
        #        
        self.assertNotEqual(afwDetect.cast_HeavyFootprint(hfoot, self.mi), None,
                            "Cast to the right sort of HeavyFootprint")
        self.assertNotEqual(afwDetect.HeavyFootprintF.cast(hfoot), None,
                            "Cast to the right sort of HeavyFootprint")

        self.assertEqual(afwDetect.cast_HeavyFootprint(self.foot, self.mi), None,
                         "Can't cast a Footprint to a HeavyFootprint")
        self.assertEqual(afwDetect.HeavyFootprintI.cast(hfoot), None,
                         "Cast to the wrong sort of HeavyFootprint")

    def testMergeHeavyFootprints(self):
        mi = afwImage.MaskedImageF(20, 10)
        objectPixelVal = (42, 0x9, 400)
        
        foot = afwDetect.Footprint()
        for y, x0, x1 in [(1, 9, 12),
                          (2, 12, 13),
                          (3, 11, 15)]:
            foot.addSpan(y, x0, x1)
            for x in range(x0, x1 + 1):
                mi.set(x, y, objectPixelVal)

        hfoot1 = afwDetect.makeHeavyFootprint(self.foot, self.mi)
        hfoot2 = afwDetect.makeHeavyFootprint(foot, mi)

        hsum = afwDetect.mergeHeavyFootprintsF(hfoot1, hfoot2)
        
        bb = hsum.getBBox()
        self.assertEquals(bb.getMinX(), 9)
        self.assertEquals(bb.getMaxX(), 15)
        self.assertEquals(bb.getMinY(), 1)
        self.assertEquals(bb.getMaxY(), 3)

        msum = afwImage.MaskedImageF(20,10)
        hsum.insert(msum)

        sa = msum.getImage().getArray()

        self.assertTrue(np.all(sa[1, 9:13] == objectPixelVal[0]))
        self.assertTrue(np.all(sa[2, 12:14] == objectPixelVal[0] + self.objectPixelVal[0]))
        self.assertTrue(np.all(sa[2, 10:12] == self.objectPixelVal[0]))

        sv = msum.getVariance().getArray()

        self.assertTrue(np.all(sv[1, 9:13] == objectPixelVal[2]))
        self.assertTrue(np.all(sv[2, 12:14] == objectPixelVal[2] + self.objectPixelVal[2]))
        self.assertTrue(np.all(sv[2, 10:12] == self.objectPixelVal[2]))

        sm = msum.getMask().getArray()

        self.assertTrue(np.all(sm[1, 9:13] == objectPixelVal[1]))
        self.assertTrue(np.all(sm[2, 12:14] == objectPixelVal[1] | self.objectPixelVal[1]))
        self.assertTrue(np.all(sm[2, 10:12] == self.objectPixelVal[1]))
        

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(HeavyFootprintTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests"""
    tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
