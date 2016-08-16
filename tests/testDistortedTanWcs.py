#!/usr/bin/env python
from __future__ import absolute_import, division
from builtins import range

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
import unittest

from lsst.afw.cameraGeom import TAN_PIXELS
from lsst.afw.cameraGeom.testUtils import DetectorWrapper
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
from lsst.afw.image.utils import getDistortedWcs
import lsst.utils.tests as utilsTests
import lsst.daf.base as dafBase

try:
    type(verbose)
except NameError:
    verbose = 0


class DistortedTanWcsTestCase(unittest.TestCase):
    """Test that makeWcs correctly returns a Wcs or TanWcs object
       as appropriate based on the contents of a fits header
    """

    def setUp(self):
        # metadata taken from CFHT data
        # v695856-e0/v695856-e0-c000-a00.sci_img.fits

        metadata = dafBase.PropertySet()

        metadata.set("RADECSYS", 'FK5')
        metadata.set("EQUINOX", 2000.)
        metadata.setDouble("CRVAL1", 215.604025685476)
        metadata.setDouble("CRVAL2", 53.1595451514076)
        metadata.setDouble("CRPIX1", 1109.99981456774)
        metadata.setDouble("CRPIX2", 560.018167811613)
        metadata.set("CTYPE1", "RA---TAN")
        metadata.set("CTYPE2", "DEC--TAN")
        metadata.setDouble("CD1_1", 5.10808596133527E-05)
        metadata.setDouble("CD1_2", 1.85579539217196E-07)
        metadata.setDouble("CD2_2", -5.10281493481982E-05)
        metadata.setDouble("CD2_1", -8.27440751733828E-07)
        self.tanWcs = afwImage.cast_TanWcs(afwImage.makeWcs(metadata))

    def tearDown(self):
        del self.tanWcs

    def testBasics(self):
        pixelsToTanPixels = afwGeom.RadialXYTransform([0, 1.001, 0.00003])
        distortedWcs = afwImage.DistortedTanWcs(self.tanWcs, pixelsToTanPixels)
        tanWcsCopy = distortedWcs.getTanWcs()

        self.assertEqual(self.tanWcs, tanWcsCopy)
        self.assertFalse(self.tanWcs.hasDistortion())
        self.assertTrue(distortedWcs.hasDistortion())
        try:
            self.tanWcs == distortedWcs
            self.fail("== should not be implemented for DistortedTanWcs")
        except Exception:
            pass
        try:
            distortedWcs == self.tanWcs
            self.fail("== should not be implemented for DistortedTanWcs")
        except Exception:
            pass

    def testTransform(self):
        """Test pixelToSky, skyToPixel, getTanWcs and getPixelToTanPixel
        """
        pixelsToTanPixels = afwGeom.RadialXYTransform([0, 1.001, 0.00003])
        distortedWcs = afwImage.DistortedTanWcs(self.tanWcs, pixelsToTanPixels)
        tanWcsCopy = distortedWcs.getTanWcs()
        pixToTanCopy = distortedWcs.getPixelToTanPixel()

        for x in (0, 1000, 5000):
            for y in (0, 560, 2000):
                pixPos = afwGeom.Point2D(x, y)
                tanPixPos = pixelsToTanPixels.forwardTransform(pixPos)

                tanPixPosCopy = pixToTanCopy.forwardTransform(pixPos)
                self.assertEqual(tanPixPos, tanPixPosCopy)

                predSky = self.tanWcs.pixelToSky(tanPixPos)
                predSkyCopy = tanWcsCopy.pixelToSky(tanPixPos)
                self.assertEqual(predSky, predSkyCopy)

                measSky = distortedWcs.pixelToSky(pixPos)
                self.assertLess(predSky.angularSeparation(measSky).asRadians(), 1e-7)

                pixPosRoundTrip = distortedWcs.skyToPixel(measSky)
                for i in range(2):
                    self.assertAlmostEqual(pixPos[i], pixPosRoundTrip[i])

    def testGetDistortedWcs(self):
        """Test utils.getDistortedWcs
        """
        dw = DetectorWrapper()
        detector = dw.detector

        # the standard case: the exposure's WCS is pure TAN WCS and distortion information is available;
        # return a DistortedTanWcs
        exposure = afwImage.ExposureF(10, 10)
        exposure.setDetector(detector)
        exposure.setWcs(self.tanWcs)
        self.assertFalse(self.tanWcs.hasDistortion())
        outWcs = getDistortedWcs(exposure.getInfo())
        self.assertTrue(outWcs.hasDistortion())
        self.assertTrue(afwImage.DistortedTanWcs.cast(outWcs) is not None)
        del exposure  # avoid accidental reuse
        del outWcs

        # return the original WCS if the exposure's WCS has distortion
        pixelsToTanPixels = afwGeom.RadialXYTransform([0, 1.001, 0.00003])
        distortedWcs = afwImage.DistortedTanWcs(self.tanWcs, pixelsToTanPixels)
        self.assertTrue(distortedWcs.hasDistortion())
        exposure = afwImage.ExposureF(10, 10)
        exposure.setWcs(distortedWcs)
        exposure.setDetector(detector)
        outWcs = getDistortedWcs(exposure.getInfo())
        self.assertTrue(outWcs.hasDistortion())
        self.assertTrue(afwImage.DistortedTanWcs.cast(outWcs) is not None)
        del exposure
        del distortedWcs
        del outWcs

        # raise an exception if exposure has no WCS
        exposure = afwImage.ExposureF(10, 10)
        exposure.setDetector(detector)
        self.assertRaises(Exception, getDistortedWcs, exposure.getInfo())
        del exposure

        # return the original pure TAN WCS if the exposure has no detector
        exposure = afwImage.ExposureF(10, 10)
        exposure.setWcs(self.tanWcs)
        outWcs = getDistortedWcs(exposure.getInfo())
        self.assertFalse(outWcs.hasDistortion())
        self.assertTrue(afwImage.TanWcs.cast(outWcs) is not None)
        self.assertTrue(afwImage.DistortedTanWcs.cast(outWcs) is None)
        del exposure
        del outWcs

        # return the original pure TAN WCS if the exposure's detector has no TAN_PIXELS transform
        def removeTanPixels(detectorWrapper):
            tanPixSys = detector.makeCameraSys(TAN_PIXELS)
            detectorWrapper.transMap.pop(tanPixSys)
        detectorNoTanPix = DetectorWrapper(modFunc=removeTanPixels).detector
        exposure = afwImage.ExposureF(10, 10)
        exposure.setWcs(self.tanWcs)
        exposure.setDetector(detectorNoTanPix)
        outWcs = getDistortedWcs(exposure.getInfo())
        self.assertFalse(outWcs.hasDistortion())
        self.assertTrue(afwImage.TanWcs.cast(outWcs) is not None)
        self.assertTrue(afwImage.DistortedTanWcs.cast(outWcs) is None)
        del exposure
        del outWcs


def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(DistortedTanWcsTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
