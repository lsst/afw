#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008-2014 LSST Corporation.
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
#pybind11#Tests for detection.GaussianPsf
#pybind11#
#pybind11#Run with:
#pybind11#   ./testGaussianPsf.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import testGaussianPsf; testGaussianPsf.run()
#pybind11#"""
#pybind11#
#pybind11#import unittest
#pybind11#import numpy
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.afw.table
#pybind11#import lsst.afw.geom
#pybind11#import lsst.afw.coord
#pybind11#import lsst.afw.fits
#pybind11#import lsst.afw.detection
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11#
#pybind11#def makeGaussianImage(bbox, sigma, xc=0.0, yc=0.0):
#pybind11#    image = lsst.afw.image.ImageD(bbox)
#pybind11#    array = image.getArray()
#pybind11#    for yi, yv in enumerate(range(bbox.getBeginY(), bbox.getEndY())):
#pybind11#        for xi, xv in enumerate(range(bbox.getBeginX(), bbox.getEndX())):
#pybind11#            array[yi, xi] = numpy.exp(-0.5*((xv - xc)**2 + (yv - yc)**2)/sigma**2)
#pybind11#    array /= array.sum()
#pybind11#    return image
#pybind11#
#pybind11#
#pybind11#def computeNaiveApertureFlux(image, radius, xc=0.0, yc=0.0):
#pybind11#    bbox = image.getBBox()
#pybind11#    array = image.getArray()
#pybind11#    s = 0.0
#pybind11#    for yi, yv in enumerate(range(bbox.getBeginY(), bbox.getEndY())):
#pybind11#        for xi, xv in enumerate(range(bbox.getBeginX(), bbox.getEndX())):
#pybind11#            if (xv - xc)**2 + (yv - yc)**2 < radius**2:
#pybind11#                s += array[yi, xi]
#pybind11#    return s
#pybind11#
#pybind11#
#pybind11#class GaussianPsfTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.psf = lsst.afw.detection.GaussianPsf(51, 51, 4.0)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.psf
#pybind11#
#pybind11#    def testKernelImage(self):
#pybind11#        image = self.psf.computeKernelImage()
#pybind11#        check = makeGaussianImage(image.getBBox(), self.psf.getSigma())
#pybind11#        self.assertFloatsAlmostEqual(image.getArray(), check.getArray())
#pybind11#        self.assertFloatsAlmostEqual(image.getArray().sum(), 1.0, atol=1E-14)
#pybind11#
#pybind11#    def testOffsetImage(self):
#pybind11#        image = self.psf.computeImage(lsst.afw.geom.Point2D(0.25, 0.25))
#pybind11#        check = makeGaussianImage(image.getBBox(), self.psf.getSigma(), 0.25, 0.25)
#pybind11#        self.assertFloatsAlmostEqual(image.getArray(), check.getArray(), atol=1E-4, rtol=1E-4, plotOnFailure=True)
#pybind11#
#pybind11#    def testApertureFlux(self):
#pybind11#        image = self.psf.computeKernelImage(lsst.afw.geom.Point2D(0.0, 0.0))
#pybind11#        # test aperture implementation is very crude; can only test to about 10%
#pybind11#        self.assertFloatsAlmostEqual(self.psf.computeApertureFlux(5.0), computeNaiveApertureFlux(image, 5.0), rtol=0.1)
#pybind11#        self.assertFloatsAlmostEqual(self.psf.computeApertureFlux(7.0), computeNaiveApertureFlux(image, 7.0), rtol=0.1)
#pybind11#
#pybind11#    def testShape(self):
#pybind11#        self.assertFloatsAlmostEqual(self.psf.computeShape().getDeterminantRadius(), 4.0)
#pybind11#
#pybind11#    def testPersistence(self):
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as filename:
#pybind11#            self.psf.writeFits(filename)
#pybind11#            psf = lsst.afw.detection.GaussianPsf.readFits(filename)
#pybind11#            self.assertEqual(self.psf.getSigma(), psf.getSigma())
#pybind11#            self.assertEqual(self.psf.getDimensions(), psf.getDimensions())
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
