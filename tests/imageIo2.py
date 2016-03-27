#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

import unittest
import lsst.utils.tests as utilsTests
import lsst.afw.image as afwImage

class ImageIoTestCase(unittest.TestCase):
    """A test case for Image Persistence"""

    def checkImages(self, image, original):
        # Check that two images are identical
        self.assertEqual(image.getHeight(), original.getHeight())
        self.assertEqual(image.getWidth(), original.getWidth())
        self.assertEqual(image.getY0(), original.getY0())
        self.assertEqual(image.getX0(), original.getX0())
        for x in xrange(0, original.getWidth()):
            for y in xrange(0, image.getHeight()):
                self.assertEqual(image.get(x, y), original.get(x, y))

    def setUp(self):
        # Create the additionalData PropertySet
        self.cols = 4
        self.rows = 4

    def testIo(self):
        for Image in (afwImage.ImageU,
                      afwImage.ImageL,
                      afwImage.ImageI,
                      afwImage.ImageF,
                      afwImage.ImageD,
                      ):
            image = Image(self.cols, self.rows)
            for x in xrange(0, self.cols):
                for y in xrange(0, self.rows):
                    image.set(x, y, x + y)

            with utilsTests.getTempFilePath("_%s.fits" % (Image.__name__,)) as filename:
                image.writeFits(filename)
                readImage = Image(filename)

            self.checkImages(readImage, image)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ImageIoTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
