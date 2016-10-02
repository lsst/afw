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
#pybind11#Tests for HeavyFootprints
#pybind11#
#pybind11#Run with:
#pybind11#   heavyFootprint.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import heavyFootprint; heavyFootprint.run()
#pybind11#"""
#pybind11#
#pybind11#import numpy as np
#pybind11#import os
#pybind11#import unittest
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.detection as afwDetect
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#from lsst.log import Log
#pybind11#
#pybind11#Log.getLogger("afw.image.Mask").setLevel(Log.INFO)
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class HeavyFootprintTestCase(lsst.utils.tests.TestCase):
#pybind11#    """A test case for HeavyFootprint"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.mi = afwImage.MaskedImageF(20, 10)
#pybind11#        self.objectPixelVal = (10, 0x1, 100)
#pybind11#
#pybind11#        self.foot = afwDetect.Footprint()
#pybind11#        for y, x0, x1 in [(2, 10, 13),
#pybind11#                          (3, 11, 14)]:
#pybind11#            self.foot.addSpan(y, x0, x1)
#pybind11#
#pybind11#            for x in range(x0, x1 + 1):
#pybind11#                self.mi.set(x, y, self.objectPixelVal)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.foot
#pybind11#        del self.mi
#pybind11#
#pybind11#    def testCreate(self):
#pybind11#        """Check that we can create a HeavyFootprint"""
#pybind11#
#pybind11#        imi = self.mi.Factory(self.mi, True)  # copy of input image
#pybind11#
#pybind11#        hfoot = afwDetect.makeHeavyFootprint(self.foot, self.mi)
#pybind11#        self.assertNotEqual(hfoot.getId(), None)  # check we can call a base-class method
#pybind11#        #
#pybind11#        # Check we didn't modify the input image
#pybind11#        #
#pybind11#        self.assertFloatsEqual(self.mi.getImage().getArray(), imi.getImage().getArray())
#pybind11#
#pybind11#        omi = self.mi.Factory(self.mi.getDimensions())
#pybind11#        omi.set((1, 0x4, 0.1))
#pybind11#        hfoot.insert(omi)
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(imi, frame=0, title="input")
#pybind11#            ds9.mtv(omi, frame=1, title="output")
#pybind11#
#pybind11#        for s in self.foot.getSpans():
#pybind11#            y = s.getY()
#pybind11#            for x in range(s.getX0(), s.getX1() + 1):
#pybind11#                self.assertEqual(imi.get(x, y), omi.get(x, y))
#pybind11#
#pybind11#        # Check that we can call getImageArray(), etc
#pybind11#        arr = hfoot.getImageArray()
#pybind11#        print(arr)
#pybind11#        # Check that it's iterable
#pybind11#        for x in arr:
#pybind11#            pass
#pybind11#        arr = hfoot.getMaskArray()
#pybind11#        print(arr)
#pybind11#        for x in arr:
#pybind11#            pass
#pybind11#        arr = hfoot.getVarianceArray()
#pybind11#        print(arr)
#pybind11#        # Check that it's iterable
#pybind11#        for x in arr:
#pybind11#            pass
#pybind11#
#pybind11#    def testSetFootprint(self):
#pybind11#        """Check that we can create a HeavyFootprint and set the pixels under it"""
#pybind11#
#pybind11#        ctrl = afwDetect.HeavyFootprintCtrl()
#pybind11#        ctrl.setModifySource(afwDetect.HeavyFootprintCtrl.SET)  # clear the pixels in the Footprint
#pybind11#        ctrl.setMaskVal(self.objectPixelVal[1])
#pybind11#
#pybind11#        afwDetect.makeHeavyFootprint(self.foot, self.mi, ctrl)
#pybind11#        #
#pybind11#        # Check that we cleared all the pixels
#pybind11#        #
#pybind11#        self.assertEqual(np.min(self.mi.getImage().getArray()), 0.0)
#pybind11#        self.assertEqual(np.max(self.mi.getImage().getArray()), 0.0)
#pybind11#        self.assertEqual(np.min(self.mi.getMask().getArray()), 0.0)
#pybind11#        self.assertEqual(np.max(self.mi.getMask().getArray()), 0.0)
#pybind11#        self.assertEqual(np.min(self.mi.getVariance().getArray()), 0.0)
#pybind11#        self.assertEqual(np.max(self.mi.getVariance().getArray()), 0.0)
#pybind11#
#pybind11#    def testMakeHeavy(self):
#pybind11#        """Test that we can make a FootprintSet heavy"""
#pybind11#        fs = afwDetect.FootprintSet(self.mi, afwDetect.Threshold(1))
#pybind11#
#pybind11#        ctrl = afwDetect.HeavyFootprintCtrl(afwDetect.HeavyFootprintCtrl.NONE)
#pybind11#        fs.makeHeavy(self.mi, ctrl)
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(self.mi, frame=0, title="input")
#pybind11#            #ds9.mtv(omi, frame=1, title="output")
#pybind11#
#pybind11#        omi = self.mi.Factory(self.mi.getDimensions())
#pybind11#
#pybind11#        for foot in fs.getFootprints():
#pybind11#            self.assertNotEqual(afwDetect.cast_HeavyFootprint(foot, self.mi), None)
#pybind11#            afwDetect.cast_HeavyFootprint(foot, self.mi).insert(omi)
#pybind11#
#pybind11#        for foot in fs.getFootprints():
#pybind11#            self.assertNotEqual(afwDetect.HeavyFootprintF.cast(foot), None)
#pybind11#            afwDetect.HeavyFootprintF.cast(foot).insert(omi)
#pybind11#
#pybind11#        self.assertFloatsEqual(self.mi.getImage().getArray(), omi.getImage().getArray())
#pybind11#
#pybind11#    def testXY0(self):
#pybind11#        """Test that inserting a HeavyFootprint obeys XY0"""
#pybind11#        fs = afwDetect.FootprintSet(self.mi, afwDetect.Threshold(1))
#pybind11#
#pybind11#        fs.makeHeavy(self.mi)
#pybind11#
#pybind11#        bbox = afwGeom.BoxI(afwGeom.PointI(9, 1), afwGeom.ExtentI(7, 4))
#pybind11#        omi = self.mi.Factory(self.mi, bbox, afwImage.LOCAL, True)
#pybind11#        omi.set((0, 0x0, 0))
#pybind11#
#pybind11#        for foot in fs.getFootprints():
#pybind11#            afwDetect.cast_HeavyFootprint(foot, self.mi).insert(omi)
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(self.mi, frame=0, title="input")
#pybind11#            ds9.mtv(omi, frame=1, title="sub")
#pybind11#
#pybind11#        submi = self.mi.Factory(self.mi, bbox, afwImage.LOCAL)
#pybind11#        self.assertFloatsEqual(submi.getImage().getArray(), omi.getImage().getArray())
#pybind11#
#pybind11#    def testCast_HeavyFootprint(self):
#pybind11#        """Test that we can cast a Footprint to a HeavyFootprint"""
#pybind11#
#pybind11#        hfoot = afwDetect.makeHeavyFootprint(self.foot, self.mi)
#pybind11#
#pybind11#        ctrl = afwDetect.HeavyFootprintCtrl(afwDetect.HeavyFootprintCtrl.NONE)
#pybind11#        hfoot = afwDetect.makeHeavyFootprint(self.foot, self.mi, ctrl)
#pybind11#        #
#pybind11#        # This isn't quite a full test, as hfoot is already a HeavyFootprint,
#pybind11#        # the complete test is in testMakeHeavy
#pybind11#        #
#pybind11#        self.assertNotEqual(afwDetect.cast_HeavyFootprint(hfoot, self.mi), None,
#pybind11#                            "Cast to the right sort of HeavyFootprint")
#pybind11#        self.assertNotEqual(afwDetect.HeavyFootprintF.cast(hfoot), None,
#pybind11#                            "Cast to the right sort of HeavyFootprint")
#pybind11#
#pybind11#        self.assertEqual(afwDetect.cast_HeavyFootprint(self.foot, self.mi), None,
#pybind11#                         "Can't cast a Footprint to a HeavyFootprint")
#pybind11#        self.assertEqual(afwDetect.HeavyFootprintI.cast(hfoot), None,
#pybind11#                         "Cast to the wrong sort of HeavyFootprint")
#pybind11#
#pybind11#    def testMergeHeavyFootprints(self):
#pybind11#        mi = afwImage.MaskedImageF(20, 10)
#pybind11#        objectPixelVal = (42, 0x9, 400)
#pybind11#
#pybind11#        foot = afwDetect.Footprint()
#pybind11#        for y, x0, x1 in [(1, 9, 12),
#pybind11#                          (2, 12, 13),
#pybind11#                          (3, 11, 15)]:
#pybind11#            foot.addSpan(y, x0, x1)
#pybind11#            for x in range(x0, x1 + 1):
#pybind11#                mi.set(x, y, objectPixelVal)
#pybind11#
#pybind11#        hfoot1 = afwDetect.makeHeavyFootprint(self.foot, self.mi)
#pybind11#        hfoot2 = afwDetect.makeHeavyFootprint(foot, mi)
#pybind11#
#pybind11#        hfoot1.normalize()
#pybind11#        hfoot2.normalize()
#pybind11#        hsum = afwDetect.mergeHeavyFootprintsF(hfoot1, hfoot2)
#pybind11#
#pybind11#        bb = hsum.getBBox()
#pybind11#        self.assertEquals(bb.getMinX(), 9)
#pybind11#        self.assertEquals(bb.getMaxX(), 15)
#pybind11#        self.assertEquals(bb.getMinY(), 1)
#pybind11#        self.assertEquals(bb.getMaxY(), 3)
#pybind11#
#pybind11#        msum = afwImage.MaskedImageF(20, 10)
#pybind11#        hsum.insert(msum)
#pybind11#
#pybind11#        sa = msum.getImage().getArray()
#pybind11#
#pybind11#        self.assertFloatsEqual(sa[1, 9:13], objectPixelVal[0])
#pybind11#        self.assertFloatsEqual(sa[2, 12:14], objectPixelVal[0] + self.objectPixelVal[0])
#pybind11#        self.assertFloatsEqual(sa[2, 10:12], self.objectPixelVal[0])
#pybind11#
#pybind11#        sv = msum.getVariance().getArray()
#pybind11#
#pybind11#        self.assertFloatsEqual(sv[1, 9:13], objectPixelVal[2])
#pybind11#        self.assertFloatsEqual(sv[2, 12:14], objectPixelVal[2] + self.objectPixelVal[2])
#pybind11#        self.assertFloatsEqual(sv[2, 10:12], self.objectPixelVal[2])
#pybind11#
#pybind11#        sm = msum.getMask().getArray()
#pybind11#
#pybind11#        self.assertFloatsEqual(sm[1, 9:13], objectPixelVal[1])
#pybind11#        self.assertFloatsEqual(sm[2, 12:14], objectPixelVal[1] | self.objectPixelVal[1])
#pybind11#        self.assertFloatsEqual(sm[2, 10:12], self.objectPixelVal[1])
#pybind11#
#pybind11#        if False:
#pybind11#            import matplotlib
#pybind11#            matplotlib.use('Agg')
#pybind11#            import pylab as plt
#pybind11#            im1 = afwImage.ImageF(bb)
#pybind11#            hfoot1.insert(im1)
#pybind11#            im2 = afwImage.ImageF(bb)
#pybind11#            hfoot2.insert(im2)
#pybind11#            im3 = afwImage.ImageF(bb)
#pybind11#            hsum.insert(im3)
#pybind11#            plt.clf()
#pybind11#            plt.subplot(1, 3, 1)
#pybind11#            plt.imshow(im1.getArray(), interpolation='nearest', origin='lower')
#pybind11#            plt.subplot(1, 3, 2)
#pybind11#            plt.imshow(im2.getArray(), interpolation='nearest', origin='lower')
#pybind11#            plt.subplot(1, 3, 3)
#pybind11#            plt.imshow(im3.getArray(), interpolation='nearest', origin='lower')
#pybind11#            plt.savefig('merge.png')
#pybind11#
#pybind11#    def testFitsPersistence(self):
#pybind11#        heavy1 = afwDetect.HeavyFootprintF(self.foot)
#pybind11#        heavy1.getImageArray()[:] = np.random.randn(self.foot.getArea()).astype(np.float32)
#pybind11#        heavy1.getMaskArray()[:] = np.random.randint(low=0, high=2,
#pybind11#                                                     size=self.foot.getArea()).astype(np.uint16)
#pybind11#        heavy1.getVarianceArray()[:] = np.random.randn(self.foot.getArea()).astype(np.float32)
#pybind11#        filename = "heavyFootprint-testFitsPersistence.fits"
#pybind11#        heavy1.writeFits(filename)
#pybind11#        heavy2 = afwDetect.HeavyFootprintF.readFits(filename)
#pybind11#        self.assertEqual(heavy1.getArea(), heavy2.getArea())
#pybind11#        self.assertEqual(list(heavy1.getSpans()), list(heavy2.getSpans()))
#pybind11#        self.assertEqual(list(heavy1.getPeaks()), list(heavy2.getPeaks()))
#pybind11#        self.assertClose(heavy1.getImageArray(), heavy2.getImageArray(), rtol=0.0, atol=0.0)
#pybind11#        self.assertClose(heavy1.getMaskArray(), heavy2.getMaskArray(), rtol=0.0, atol=0.0)
#pybind11#        self.assertClose(heavy1.getVarianceArray(), heavy2.getVarianceArray(), rtol=0.0, atol=0.0)
#pybind11#        os.remove(filename)
#pybind11#
#pybind11#    def testDot(self):
#pybind11#        """Test HeavyFootprint::dot"""
#pybind11#        size = 20, 20
#pybind11#        for xOffset, yOffset in [(0, 0), (0, 3), (3, 0), (2, 2)]:
#pybind11#            mi1 = afwImage.MaskedImageF(*size)
#pybind11#            mi2 = afwImage.MaskedImageF(*size)
#pybind11#            mi1.set(0)
#pybind11#            mi2.set(0)
#pybind11#
#pybind11#            fp1 = afwDetect.Footprint()
#pybind11#            fp2 = afwDetect.Footprint()
#pybind11#            for y, x0, x1 in [(5, 3, 7),
#pybind11#                              (6, 3, 4),
#pybind11#                              (6, 6, 7),
#pybind11#                              (7, 3, 7), ]:
#pybind11#                fp1.addSpan(y, x0, x1)
#pybind11#                fp2.addSpan(y + yOffset, x0 + xOffset, x1 + xOffset)
#pybind11#                for x in range(x0, x1 + 1):
#pybind11#                    value = (x + y, 0, 1.0)
#pybind11#                    mi1.set(x, y, value)
#pybind11#                    mi2.set(x + xOffset, y + yOffset, value)
#pybind11#
#pybind11#            hfp1 = afwDetect.makeHeavyFootprint(fp1, mi1)
#pybind11#            hfp2 = afwDetect.makeHeavyFootprint(fp2, mi2)
#pybind11#            hfp1.normalize()
#pybind11#            hfp2.normalize()
#pybind11#
#pybind11#            dot = np.vdot(mi1.getImage().getArray(), mi2.getImage().getArray())
#pybind11#            self.assertEqual(hfp1.dot(hfp2), dot)
#pybind11#            self.assertEqual(hfp2.dot(hfp1), dot)
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
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
