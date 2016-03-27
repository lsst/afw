#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

"""
Tests for detection.GaussianPsf

Run with:
   ./testGaussianPsf.py
or
   python
   >>> import testGaussianPsf; testGaussianPsf.run()
"""

import unittest
import numpy

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.table
import lsst.afw.geom
import lsst.afw.coord
import lsst.afw.fits
import lsst.afw.detection

try:
    type(display)
except NameError:
    display = False

def makeGaussianImage(bbox, sigma, xc=0.0, yc=0.0):
    image = lsst.afw.image.ImageD(bbox)
    array = image.getArray()
    for yi, yv in enumerate(xrange(bbox.getBeginY(), bbox.getEndY())):
        for xi, xv in enumerate(xrange(bbox.getBeginX(), bbox.getEndX())):
            array[yi, xi] = numpy.exp(-0.5*((xv - xc)**2 + (yv - yc)**2)/sigma**2)
    array /= array.sum()
    return image

def computeNaiveApertureFlux(image, radius, xc=0.0, yc=0.0):
    bbox = image.getBBox()
    array = image.getArray()
    s = 0.0
    for yi, yv in enumerate(xrange(bbox.getBeginY(), bbox.getEndY())):
        for xi, xv in enumerate(xrange(bbox.getBeginX(), bbox.getEndX())):
            if (xv - xc)**2 + (yv - yc)**2 < radius**2:
                s += array[yi, xi]
    return s

class GaussianPsfTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        self.psf = lsst.afw.detection.GaussianPsf(51, 51, 4.0)

    def tearDown(self):
        del self.psf

    def testKernelImage(self):
        image = self.psf.computeKernelImage()
        check = makeGaussianImage(image.getBBox(), self.psf.getSigma())
        self.assertClose(image.getArray(), check.getArray())
        self.assertClose(image.getArray().sum(), 1.0, atol=1E-14)

    def testOffsetImage(self):
        image = self.psf.computeImage(lsst.afw.geom.Point2D(0.25, 0.25))
        check = makeGaussianImage(image.getBBox(), self.psf.getSigma(), 0.25, 0.25)
        self.assertClose(image.getArray(), check.getArray(), atol=1E-4, rtol=1E-4, plotOnFailure=True)

    def testApertureFlux(self):
        image = self.psf.computeKernelImage(lsst.afw.geom.Point2D(0.0, 0.0))
        # test aperture implementation is very crude; can only test to about 10%
        self.assertClose(self.psf.computeApertureFlux(5.0), computeNaiveApertureFlux(image, 5.0), rtol=0.1)
        self.assertClose(self.psf.computeApertureFlux(7.0), computeNaiveApertureFlux(image, 7.0), rtol=0.1)

    def testShape(self):
        self.assertClose(self.psf.computeShape().getDeterminantRadius(), 4.0)

    def testPersistence(self):
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            self.psf.writeFits(filename)
            psf = lsst.afw.detection.GaussianPsf.readFits(filename)
            self.assertEqual(self.psf.getSigma(), psf.getSigma())
            self.assertEqual(self.psf.getDimensions(), psf.getDimensions())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(GaussianPsfTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
