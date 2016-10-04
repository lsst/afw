#!/usr/bin/env python
from __future__ import absolute_import, division
from builtins import range

#
# LSST Data Management System
# Copyright 2008-2014 LSST Corporation.
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
    for yi, yv in enumerate(range(bbox.getBeginY(), bbox.getEndY())):
        for xi, xv in enumerate(range(bbox.getBeginX(), bbox.getEndX())):
            array[yi, xi] = numpy.exp(-0.5*((xv - xc)**2 + (yv - yc)**2)/sigma**2)
    array /= array.sum()
    return image


def computeNaiveApertureFlux(image, radius, xc=0.0, yc=0.0):
    bbox = image.getBBox()
    array = image.getArray()
    s = 0.0
    for yi, yv in enumerate(range(bbox.getBeginY(), bbox.getEndY())):
        for xi, xv in enumerate(range(bbox.getBeginX(), bbox.getEndX())):
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
        self.assertFloatsAlmostEqual(image.getArray(), check.getArray())
        self.assertFloatsAlmostEqual(image.getArray().sum(), 1.0, atol=1E-14)

    def testOffsetImage(self):
        image = self.psf.computeImage(lsst.afw.geom.Point2D(0.25, 0.25))
        check = makeGaussianImage(image.getBBox(), self.psf.getSigma(), 0.25, 0.25)
        self.assertFloatsAlmostEqual(image.getArray(), check.getArray(), atol=1E-4, rtol=1E-4,
                                     plotOnFailure=True)

    def testApertureFlux(self):
        image = self.psf.computeKernelImage(lsst.afw.geom.Point2D(0.0, 0.0))
        # test aperture implementation is very crude; can only test to about 10%
        self.assertFloatsAlmostEqual(self.psf.computeApertureFlux(5.0), computeNaiveApertureFlux(image, 5.0),
                                     rtol=0.1)
        self.assertFloatsAlmostEqual(self.psf.computeApertureFlux(7.0), computeNaiveApertureFlux(image, 7.0),
                                     rtol=0.1)

    def testShape(self):
        self.assertFloatsAlmostEqual(self.psf.computeShape().getDeterminantRadius(), 4.0)

    def testPersistence(self):
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            self.psf.writeFits(filename)
            psf = lsst.afw.detection.GaussianPsf.readFits(filename)
            self.assertEqual(self.psf.getSigma(), psf.getSigma())
            self.assertEqual(self.psf.getDimensions(), psf.getDimensions())


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
