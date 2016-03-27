#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

import unittest
import pickle

import numpy

import lsst.utils.tests as utilsTests
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage

class ImagePickleTestCase(unittest.TestCase):
    """A test case for Image pickling"""
    def setUp(self):
        self.xSize = 4
        self.ySize = 7
        self.x0 = 567
        self.y0 = 98765

    def createImage(self, factory=afwImage.ImageF):
        image = factory(self.xSize, self.ySize)
        image.setXY0(afwGeom.Point2I(self.x0, self.y0))
        image.getArray()[:] = self.createPattern()
        return image

    def createMaskedImage(self, factory=afwImage.MaskedImageF):
        image = factory(self.xSize, self.ySize)
        image.setXY0(afwGeom.Point2I(self.x0, self.y0))
        image.getImage().getArray()[:] = self.createPattern()
        image.getMask().getArray()[:] = self.createPattern()
        image.getVariance().getArray()[:] = self.createPattern()
        return image

    def createPattern(self):
        yy, xx = numpy.ogrid[0:self.ySize, 0:self.xSize] # NB: numpy operates 'backwards'
        return self.xSize*yy + xx

    def assertImagesEqual(self, image, original):
        self.assertEqual(image.__class__.__name__, original.__class__.__name__)
        self.assertEqual(image.getHeight(), original.getHeight())
        self.assertEqual(image.getWidth(), original.getWidth())
        self.assertEqual(image.getY0(), original.getY0())
        self.assertEqual(image.getX0(), original.getX0())
        for x in xrange(0, original.getWidth()):
            for y in xrange(0, image.getHeight()):
                self.assertEqual(image.get(x, y), original.get(x, y))

    def checkImages(self, original):
        image = pickle.loads(pickle.dumps(original))
        self.assertImagesEqual(image, original)

    def checkExposures(self, original):
        image = pickle.loads(pickle.dumps(original))
        self.assertImagesEqual(image.getMaskedImage(), original.getMaskedImage())
        self.assertEqual(image.getWcs(), original.getWcs())

    def testImage(self):
        for Image in (afwImage.ImageU,
                      afwImage.ImageI,
                      afwImage.ImageF,
                      afwImage.ImageD,
                      afwImage.MaskU,
                      ):
            image = self.createImage(Image)
            self.checkImages(image)

    def testMaskedImage(self):
        scale = (1.0*afwGeom.arcseconds).asDegrees()
        wcs = afwImage.makeWcs(afwCoord.Coord(0.0*afwGeom.degrees, 0.0*afwGeom.degrees),
                               afwGeom.Point2D(0.0, 0.0), scale, 0.0, 0.0, scale)
        for MaskedImage in (afwImage.MaskedImageF,
                            afwImage.MaskedImageD,
                        ):
            image = self.createMaskedImage(MaskedImage)
            self.checkImages(image)
            exposure = afwImage.makeExposure(image, wcs)
            self.checkExposures(exposure)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ImagePickleTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
