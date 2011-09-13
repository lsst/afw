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
Tests for Footprints, and FootprintSets

Run with:
   footprint2.py
or
   python
   >>> import footprint2; footprint2.run()
"""


import sys
import unittest
import lsst.utils.tests as tests
import lsst.pex.logging as logging
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDetect
import lsst.afw.detection.utils as afwDetectUtils
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

def toString(*args):
    """toString written in python"""
    if len(args) == 1:
        args = args[0]

    y, x0, x1 = args
    return "%d: %d..%d" % (y, x0, x1)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class Object(object):
    def __init__(self, val, spans):            
        self.val = val
        self.spans = spans

    def insert(self, im):
        """Insert self into an image"""
        for sp in self.spans:
            y, x0, x1 = sp
            for x in range(x0, x1+1):
                im.set(x, y, self.val)

    def __eq__(self, other):
        for osp, sp in zip(other.getSpans(), self.spans):
            if osp.toString() != toString(sp):
                return False

        return True

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class FootprintSetUTestCase(unittest.TestCase):
    """A test case for FootprintSet"""

    def setUp(self):
        self.im = afwImage.ImageU(afwGeom.Extent2I(12, 8))
        #
        # Objects that we should detect
        #
        self.objects = []
        self.objects += [Object(10, [(1, 4, 4), (2, 3, 5), (3, 4, 4)])]
        self.objects += [Object(20, [(5, 7, 8), (5, 10, 10), (6, 8, 9)])]
        self.objects += [Object(20, [(6, 3, 3)])]

        self.im.set(0)                       # clear image
        for obj in self.objects:
            obj.insert(self.im)

        if False and display:
            ds9.mtv(self.im, frame=0)
        
    def tearDown(self):
        del self.im

    def testGC(self):
        """Check that FootprintSets are automatically garbage collected (when MemoryTestCase runs)"""
        
        ds = afwDetect.FootprintSetU(afwImage.ImageU(afwGeom.Extent2I(10, 20)), afwDetect.Threshold(10))

    def testFootprints(self):
        """Check that we found the correct number of objects and that they are correct"""
        ds = afwDetect.makeFootprintSet(self.im, afwDetect.Threshold(10))

        objects = ds.getFootprints()

        self.assertEqual(len(objects), len(self.objects))
        for i in range(len(objects)):
            self.assertEqual(objects[i], self.objects[i])
            
    def testFootprints2(self):
        """Check that we found the correct number of objects using makeFootprintSet"""
        ds = afwDetect.makeFootprintSet(self.im, afwDetect.Threshold(10))

        objects = ds.getFootprints()

        self.assertEqual(len(objects), len(self.objects))
        for i in range(len(objects)):
            self.assertEqual(objects[i], self.objects[i])
            

    def testFootprintsImageId(self):
        """Check that we can insert footprints into an Image"""
        ds = afwDetect.makeFootprintSet(self.im, afwDetect.Threshold(10))
        objects = ds.getFootprints()

        idImage = afwImage.ImageU(self.im.getDimensions())
        idImage.set(0)
        
        for foot in objects:
            foot.insertIntoImage(idImage, foot.getId())

        if False:
            ds9.mtv(idImage, frame=2)

        for i in range(len(objects)):
            for sp in objects[i].getSpans():
                for x in range(sp.getX0(), sp.getX1() + 1):
                    self.assertEqual(idImage.get(x, sp.getY()), objects[i].getId())


    def testFootprintSetImageId(self):
        """Check that we can insert a FootprintSet into an Image, setting relative IDs"""
        ds = afwDetect.makeFootprintSet(self.im, afwDetect.Threshold(10))
        objects = ds.getFootprints()

        idImage = ds.insertIntoImage(True)
        if display:
            ds9.mtv(idImage, frame=2)

        for i in range(len(objects)):
            for sp in objects[i].getSpans():
                for x in range(sp.getX0(), sp.getX1() + 1):
                    self.assertEqual(idImage.get(x, sp.getY()), i + 1)

    def testFootprintsImage(self):
        """Check that we can search Images as well as MaskedImages"""
        ds = afwDetect.makeFootprintSet(self.im, afwDetect.Threshold(10))

        objects = ds.getFootprints()

        self.assertEqual(len(objects), len(self.objects))
        for i in range(len(objects)):
            self.assertEqual(objects[i], self.objects[i])
            
    def testGrow2(self):
        """Grow some more interesting shaped Footprints.  Informative with display, but no numerical tests""" 
        #Can't set mask plane as the image is not a masked image.
        ds = afwDetect.makeFootprintSet(self.im, afwDetect.Threshold(10))

        idImage = afwImage.ImageU(self.im.getDimensions())
        idImage.set(0)

        i = 1
        for foot in ds.getFootprints()[0:1]:
            gfoot = afwDetect.growFootprint(foot, 3, False)
            gfoot.insertIntoImage(idImage, i)
            i += 1

        if display:
            ds9.mtv(self.im, frame=0)
            ds9.mtv(idImage, frame=1)

    def testInf(self):
        """Test detection for images with Infs"""

        im = afwImage.MaskedImageF(afwGeom.Extent2I(10, 20))
        im.set(0)
        
        import numpy
        for x in range(im.getWidth()):
            im.set(x, im.getHeight() - 1, (numpy.Inf, 0x0, 0))

        ds = afwDetect.makeFootprintSet(im, afwDetect.createThreshold(100))

        objects = ds.getFootprints()
        afwDetect.setMaskFromFootprintList(im.getMask(), objects, 0x10)

        if display:
            ds9.mtv(im)

        self.assertEqual(len(objects), 1)
            

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class PeaksInFootprintsTestCase(unittest.TestCase):
    """A test case for detecting Peaks within Footprints"""

    def doSetUp(self, dwidth=0, dheight=0):
        self.im = afwImage.MaskedImageF(afwGeom.Extent2I(12 + dwidth, 8 + dheight))
        #
        # Objects that we should detect
        #
        self.objects, self.peaks = [], []
        self.objects.append([(4, 1, 10), (3, 2, 10), (4, 2, 20), (5, 2, 10), (4, 3, 10),])
        self.peaks.append([(4, 2)])
        self.objects.append([(7, 5, 30), (8, 5, 29), (10, 5, 25), (8, 6, 27), (9, 6, 26),])
        self.peaks.append([(7, 5)])
        self.objects.append([(3, 6, 10), (4, 6, 10),])
        self.peaks.append([(3, 6), (4, 6),])

        self.im.set((0, 0x0, 0))                       # clear image
        for obj in self.objects:
            for x, y, I in obj:
                self.im.getImage().set(x, y, I)
                
        if False and display:
            ds9.mtv(self.im, frame=0)
        
    def setUp(self):
        self.im = None

    def tearDown(self):
        del self.im

    def doTestPeaks(self, dwidth=0, dheight=0, callback=None, polarity=True):
        """Worker routine for tests
        polarity:  True if should search for +ve pixels"""
        
        self.doSetUp(dwidth, dheight)
        if not polarity:
            self.im *= -1
            
        if callback:
            callback()
            
        threshold = afwDetect.Threshold(10, afwDetect.Threshold.VALUE, polarity)
        fs = afwDetect.makeFootprintSet(self.im, threshold, "BINNED1")

        feet = fs.getFootprints()
        if display:
            ds9.mtv(self.im, frame=0)

            for i, foot in enumerate(feet):
                for p in foot.getPeaks():
                    ds9.dot("+", p.getIx(), p.getIy(), size=0.8, frame=0)
                for trueX, trueY in self.peaks[i]:
                    ds9.dot("x", trueX, trueY, size=0.8, ctype=ds9.RED, frame=0)

        for i, foot in enumerate(feet):
            npeak = None
            #
            # Peaks that touch the edge are handled differently, as only the single highest/lowest pixel
            # is treated as a Peak
            #
            if (dwidth != 0 or dheight != 0):
                if (foot.getBBox().getMinX() == 0 or foot.getBBox().getMaxX() == self.im.getWidth()  - 1 or
                    foot.getBBox().getMinY() == 0 or foot.getBBox().getMaxY() == self.im.getHeight() - 1):
                    npeak = 1

            if npeak is None:
                npeak = len(self.peaks[i])

            self.assertEqual(len(foot.getPeaks()), npeak)
                
            for j, p in enumerate(foot.getPeaks()):
                trueX, trueY = self.peaks[i][j]
                self.assertEqual((p.getIx(), p.getIy()), (trueX, trueY))

    def testSinglePeak(self):
        """Test that we can find single Peaks in Footprints"""

        self.doTestPeaks()

    def testSingleNegativePeak(self):
        """Test that we can find single Peaks in Footprints when looking for -ve detections"""

        self.doTestPeaks(polarity=False)

    def testSinglePeakAtEdge(self):
        """Test that we handle Peaks correctly at the edge"""
        
        self.doTestPeaks(dheight=-1)

    def testSingleNegativePeakAtEdge(self):
        """Test that we handle -ve Peaks correctly at the edge"""
        
        self.doTestPeaks(dheight=-1, polarity=False)

    def testMultiPeak(self):
        def callback():
            x, y = 10, 5
            self.im.getImage().set(x, y, 100)
            self.peaks[1].append((x, y))

        self.doTestPeaks(callback=callback)

    def testMultiNegativePeak(self):
        def callback():
            x, y = 10, 5
            self.im.getImage().set(x, y, -100)
            self.peaks[1].append((x, y))

        self.doTestPeaks(polarity=False, callback=callback)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    #suites += unittest.makeSuite(FootprintSetUTestCase)
    suites += unittest.makeSuite(PeaksInFootprintsTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests"""
    tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
