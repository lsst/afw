#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
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
#pybind11#import unittest
#pybind11#
#pybind11#from lsst.afw.cameraGeom import TAN_PIXELS
#pybind11#from lsst.afw.cameraGeom.testUtils import DetectorWrapper
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#from lsst.afw.image.utils import getDistortedWcs
#pybind11#import lsst.utils.tests
#pybind11#import lsst.daf.base as dafBase
#pybind11#
#pybind11#try:
#pybind11#    type(verbose)
#pybind11#except NameError:
#pybind11#    verbose = 0
#pybind11#
#pybind11#
#pybind11#class DistortedTanWcsTestCase(lsst.utils.tests.TestCase):
#pybind11#    """Test that makeWcs correctly returns a Wcs or TanWcs object
#pybind11#       as appropriate based on the contents of a fits header
#pybind11#    """
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        # metadata taken from CFHT data
#pybind11#        # v695856-e0/v695856-e0-c000-a00.sci_img.fits
#pybind11#
#pybind11#        metadata = dafBase.PropertySet()
#pybind11#
#pybind11#        metadata.set("RADECSYS", 'FK5')
#pybind11#        metadata.set("EQUINOX", 2000.)
#pybind11#        metadata.setDouble("CRVAL1", 215.604025685476)
#pybind11#        metadata.setDouble("CRVAL2", 53.1595451514076)
#pybind11#        metadata.setDouble("CRPIX1", 1109.99981456774)
#pybind11#        metadata.setDouble("CRPIX2", 560.018167811613)
#pybind11#        metadata.set("CTYPE1", "RA---TAN")
#pybind11#        metadata.set("CTYPE2", "DEC--TAN")
#pybind11#        metadata.setDouble("CD1_1", 5.10808596133527E-05)
#pybind11#        metadata.setDouble("CD1_2", 1.85579539217196E-07)
#pybind11#        metadata.setDouble("CD2_2", -5.10281493481982E-05)
#pybind11#        metadata.setDouble("CD2_1", -8.27440751733828E-07)
#pybind11#        self.tanWcs = afwImage.cast_TanWcs(afwImage.makeWcs(metadata))
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.tanWcs
#pybind11#
#pybind11#    def testBasics(self):
#pybind11#        pixelsToTanPixels = afwGeom.RadialXYTransform([0, 1.001, 0.00003])
#pybind11#        distortedWcs = afwImage.DistortedTanWcs(self.tanWcs, pixelsToTanPixels)
#pybind11#        tanWcsCopy = distortedWcs.getTanWcs()
#pybind11#
#pybind11#        self.assertEqual(self.tanWcs, tanWcsCopy)
#pybind11#        self.assertFalse(self.tanWcs.hasDistortion())
#pybind11#        self.assertTrue(distortedWcs.hasDistortion())
#pybind11#        try:
#pybind11#            self.tanWcs == distortedWcs
#pybind11#            self.fail("== should not be implemented for DistortedTanWcs")
#pybind11#        except Exception:
#pybind11#            pass
#pybind11#        try:
#pybind11#            distortedWcs == self.tanWcs
#pybind11#            self.fail("== should not be implemented for DistortedTanWcs")
#pybind11#        except Exception:
#pybind11#            pass
#pybind11#
#pybind11#    def testTransform(self):
#pybind11#        """Test pixelToSky, skyToPixel, getTanWcs and getPixelToTanPixel
#pybind11#        """
#pybind11#        pixelsToTanPixels = afwGeom.RadialXYTransform([0, 1.001, 0.00003])
#pybind11#        distortedWcs = afwImage.DistortedTanWcs(self.tanWcs, pixelsToTanPixels)
#pybind11#        tanWcsCopy = distortedWcs.getTanWcs()
#pybind11#        pixToTanCopy = distortedWcs.getPixelToTanPixel()
#pybind11#
#pybind11#        for x in (0, 1000, 5000):
#pybind11#            for y in (0, 560, 2000):
#pybind11#                pixPos = afwGeom.Point2D(x, y)
#pybind11#                tanPixPos = pixelsToTanPixels.forwardTransform(pixPos)
#pybind11#
#pybind11#                tanPixPosCopy = pixToTanCopy.forwardTransform(pixPos)
#pybind11#                self.assertEqual(tanPixPos, tanPixPosCopy)
#pybind11#
#pybind11#                predSky = self.tanWcs.pixelToSky(tanPixPos)
#pybind11#                predSkyCopy = tanWcsCopy.pixelToSky(tanPixPos)
#pybind11#                self.assertEqual(predSky, predSkyCopy)
#pybind11#
#pybind11#                measSky = distortedWcs.pixelToSky(pixPos)
#pybind11#                self.assertLess(predSky.angularSeparation(measSky).asRadians(), 1e-7)
#pybind11#
#pybind11#                pixPosRoundTrip = distortedWcs.skyToPixel(measSky)
#pybind11#                for i in range(2):
#pybind11#                    self.assertAlmostEqual(pixPos[i], pixPosRoundTrip[i])
#pybind11#
#pybind11#    def testGetDistortedWcs(self):
#pybind11#        """Test utils.getDistortedWcs
#pybind11#        """
#pybind11#        dw = DetectorWrapper()
#pybind11#        detector = dw.detector
#pybind11#
#pybind11#        # the standard case: the exposure's WCS is pure TAN WCS and distortion information is available;
#pybind11#        # return a DistortedTanWcs
#pybind11#        exposure = afwImage.ExposureF(10, 10)
#pybind11#        exposure.setDetector(detector)
#pybind11#        exposure.setWcs(self.tanWcs)
#pybind11#        self.assertFalse(self.tanWcs.hasDistortion())
#pybind11#        outWcs = getDistortedWcs(exposure.getInfo())
#pybind11#        self.assertTrue(outWcs.hasDistortion())
#pybind11#        self.assertIsNotNone(afwImage.DistortedTanWcs.cast(outWcs))
#pybind11#        del exposure  # avoid accidental reuse
#pybind11#        del outWcs
#pybind11#
#pybind11#        # return the original WCS if the exposure's WCS has distortion
#pybind11#        pixelsToTanPixels = afwGeom.RadialXYTransform([0, 1.001, 0.00003])
#pybind11#        distortedWcs = afwImage.DistortedTanWcs(self.tanWcs, pixelsToTanPixels)
#pybind11#        self.assertTrue(distortedWcs.hasDistortion())
#pybind11#        exposure = afwImage.ExposureF(10, 10)
#pybind11#        exposure.setWcs(distortedWcs)
#pybind11#        exposure.setDetector(detector)
#pybind11#        outWcs = getDistortedWcs(exposure.getInfo())
#pybind11#        self.assertTrue(outWcs.hasDistortion())
#pybind11#        self.assertIsNotNone(afwImage.DistortedTanWcs.cast(outWcs))
#pybind11#        del exposure
#pybind11#        del distortedWcs
#pybind11#        del outWcs
#pybind11#
#pybind11#        # raise an exception if exposure has no WCS
#pybind11#        exposure = afwImage.ExposureF(10, 10)
#pybind11#        exposure.setDetector(detector)
#pybind11#        with self.assertRaises(Exception):
#pybind11#            getDistortedWcs(exposure.getInfo())
#pybind11#        del exposure
#pybind11#
#pybind11#        # return the original pure TAN WCS if the exposure has no detector
#pybind11#        exposure = afwImage.ExposureF(10, 10)
#pybind11#        exposure.setWcs(self.tanWcs)
#pybind11#        outWcs = getDistortedWcs(exposure.getInfo())
#pybind11#        self.assertFalse(outWcs.hasDistortion())
#pybind11#        self.assertIsNotNone(afwImage.TanWcs.cast(outWcs))
#pybind11#        self.assertIsNone(afwImage.DistortedTanWcs.cast(outWcs))
#pybind11#        del exposure
#pybind11#        del outWcs
#pybind11#
#pybind11#        # return the original pure TAN WCS if the exposure's detector has no TAN_PIXELS transform
#pybind11#        def removeTanPixels(detectorWrapper):
#pybind11#            tanPixSys = detector.makeCameraSys(TAN_PIXELS)
#pybind11#            detectorWrapper.transMap.pop(tanPixSys)
#pybind11#        detectorNoTanPix = DetectorWrapper(modFunc=removeTanPixels).detector
#pybind11#        exposure = afwImage.ExposureF(10, 10)
#pybind11#        exposure.setWcs(self.tanWcs)
#pybind11#        exposure.setDetector(detectorNoTanPix)
#pybind11#        outWcs = getDistortedWcs(exposure.getInfo())
#pybind11#        self.assertFalse(outWcs.hasDistortion())
#pybind11#        self.assertIsNotNone(afwImage.TanWcs.cast(outWcs))
#pybind11#        self.assertIsNone(afwImage.DistortedTanWcs.cast(outWcs))
#pybind11#        del exposure
#pybind11#        del outWcs
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
