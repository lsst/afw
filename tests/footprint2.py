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
import lsst.afw.image.imageLib as afwImage
import lsst.afw.math.mathLib as afwMath
import lsst.afw.detection.detectionLib as afwDetect
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
        self.im = afwImage.ImageU(12, 8)
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
        
        ds = afwDetect.FootprintSetU(afwImage.ImageU(10, 20), afwDetect.Threshold(10))

    def testFootprints(self):
        """Check that we found the correct number of objects and that they are correct"""
        ds = afwDetect.FootprintSetU(self.im, afwDetect.Threshold(10))

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
        ds = afwDetect.FootprintSetU(self.im, afwDetect.Threshold(10))
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
        ds = afwDetect.FootprintSetU(self.im, afwDetect.Threshold(10))
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
        ds = afwDetect.FootprintSetU(self.im, afwDetect.Threshold(10))

        objects = ds.getFootprints()

        self.assertEqual(len(objects), len(self.objects))
        for i in range(len(objects)):
            self.assertEqual(objects[i], self.objects[i])
            
    def testGrow2(self):
        """Grow some more interesting shaped Footprints.  Informative with display, but no numerical tests""" 
        #Can't set mask plane as the image is not a masked image.
        ds = afwDetect.FootprintSetU(self.im, afwDetect.Threshold(10))

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

        im = afwImage.MaskedImageF(10, 20)
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

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(FootprintSetUTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests"""
    tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
