#
# LSST Data Management System
# Copyright 2017 LSST Corporation.
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
from __future__ import absolute_import, division, print_function
import math
import unittest

# from builtins import range

import numpy as np

# from lsst.afw.cameraGeom import TAN_PIXELS
# from lsst.afw.cameraGeom.testUtils import DetectorWrapper
# import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.utils.tests
from lsst.afw.coord import IcrsCoord
from lsst.afw.geom import arcseconds, degrees

try:
    type(verbose)
except NameError:
    verbose = 0


"""
Here's the plan:
- Make one or more pixelsToFocalPlane Transform (or vice-versa)
- Make at least two focalPlaneToPupil Transforms (or vice-versa):
    - One with no distortion
    - One or more with distortion
- Make one or more TAN SkyWcs;
    warning: you must make the plate scale of this SkyWcs the same as
    given by the pair of transforms created above, at focal plane 0,0
- Make a distorted SkyWcs for each pure TAN SkyWcs and each condition above
- For the focalPlaneToPupil with *no* distortion, verify that
    the distorted SkyWcs results are almost equal to the pure TAN SkyWcs results
- For the focalPlaneToPupil transforms and wcs pairs with distortion,
    verify the following:
    - For a set of pixel points on the CCD
        - transform pixel to sky using pure TAN SkyWcs
        - transform pixel to (undistorted) pupil
        - transform sky to distorted pixels using the distorted TAN WCS
        - transform distorted pixels to distorted pupil
        - check that undistorted pupil = distorted pupil (same point on the sky)
"""


def makeRotationMatrix(angle, scale):
    angleRad = angle.asRadians()
    sinAng = math.sin(angleRad)
    cosAng = math.cos(angleRad)
    return np.array([
        [cosAng, sinAng],
        [-sinAng, cosAng],
    ], dtype=float) * scale


class MakeDistortedTanWcsTestCase(lsst.utils.tests.TestCase):
    """Test lsst.afw.geom.makeDistortedTanWcs
    """

    def setUp(self):
        # define the position and size of one CCD in the focal plane
        self.pixelSizeMm = 0.024  # mm/pixel
        self.ccdOrientation = 5 * degrees  # orientation of pixels w.r.t. focal plane
        self.plateScale = 0.15 * arcseconds  # angle/pixel
        self.crpix = afwGeom.Point2D(100, 100)
        self.crval = IcrsCoord(10 * degrees, 40 * degrees)
        self.orientation = -45 * degrees
        self.scale = 1.0 * arcseconds
        # position of 0,0 pixel position in focal plane
        self.ccdPositionMm = afwGeom.Point2D(25.0, 10.0)
        self.focalPlaneToPixel = afwGeom.makeAffineTransformPoint2(
            offset = afwGeom.Extent2D(self.ccdPositionMm),
            rotation = self.ccdOrientation,
            scale = self.pixelSizeMm,
        )
        self.bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(2000, 4000))

    def testNoDistortion(self):
        """Test makeDistortedTanWcs using an affine pixelsToFocalPlane transform

        - Make one or more pixelsToFocalPlane Transform (or vice-versa)
        - Make at least two focalPlaneToPupil Transforms (or vice-versa):
            - One with no distortion
            - One or more with distortion
        - Make one or more TAN SkyWcs;
            warning: you must make the plate scale of this SkyWcs the same as
            given by the pair of transforms created above, at focal plane 0,0
        - Make a distorted SkyWcs for each pure TAN SkyWcs and each condition above
        - For the focalPlaneToPupil with *no* distortion, verify that
            the distorted SkyWcs results are almost equal to the pure TAN SkyWcs results
        """
        cdMatrix = afwGeom.makeCdMatrix(scale = self.scale, orientation = self.orientation)
        tanWcs = afwGeom.SkyWcs(crpix = self.crpix, crval = self.crval, cdMatrix = cdMatrix)
        radPerMm = self.plateScale.asRadians() * self.pixelSizeMm
        focalPlaneToPupil = afwGeom.makeAffineTransformPoint2(scale = radPerMm)
        distortedWcs = afwGeom.makeDistortedTanWcs(
            tanWcs = tanWcs,
            pixelToFocalPlane = self.focalPlaneToPixel.getInverse(),
            focalPlaneToPupil = focalPlaneToPupil,
        )
        self.assertWcsNearlyEqualOverBBox(distortedWcs, tanWcs, bbox = self.bbox)

    # def testBasics(self):
    #     pixelsToTanPixels = afwGeom.RadialXYTransform([0, 1.001, 0.00003])
    #     distortedWcs = afwImage.DistortedTanWcs(self.tanWcs, pixelsToTanPixels)
    #     tanWcsCopy = distortedWcs.getTanWcs()

    #     self.assertEqual(self.tanWcs, tanWcsCopy)
    #     self.assertFalse(self.tanWcs.hasDistortion())
    #     self.assertTrue(distortedWcs.hasDistortion())
    #     try:
    #         self.tanWcs == distortedWcs
    #         self.fail("== should not be implemented for DistortedTanWcs")
    #     except Exception:
    #         pass
    #     try:
    #         distortedWcs == self.tanWcs
    #         self.fail("== should not be implemented for DistortedTanWcs")
    #     except Exception:
    #         pass

    # def testTransform(self):
    #     """Test pixelToSky, skyToPixel, getTanWcs and getPixelToTanPixel
    #     """
    #     pixelsToTanPixels = afwGeom.RadialXYTransform([0, 1.001, 0.00003])
    #     distortedWcs = afwImage.DistortedTanWcs(self.tanWcs, pixelsToTanPixels)
    #     tanWcsCopy = distortedWcs.getTanWcs()
    #     pixToTanCopy = distortedWcs.getPixelToTanPixel()

    #     for x in (0, 1000, 5000):
    #         for y in (0, 560, 2000):
    #             pixPos = afwGeom.Point2D(x, y)
    #             tanPixPos = pixelsToTanPixels.forwardTransform(pixPos)

    #             tanPixPosCopy = pixToTanCopy.forwardTransform(pixPos)
    #             self.assertEqual(tanPixPos, tanPixPosCopy)

    #             predSky = self.tanWcs.pixelToSky(tanPixPos)
    #             predSkyCopy = tanWcsCopy.pixelToSky(tanPixPos)
    #             self.assertEqual(predSky, predSkyCopy)

    #             measSky = distortedWcs.pixelToSky(pixPos)
    #             self.assertLess(
    #                 predSky.angularSeparation(measSky).asRadians(), 1e-7)

    #             pixPosRoundTrip = distortedWcs.skyToPixel(measSky)
    #             for i in range(2):
    #                 self.assertAlmostEqual(pixPos[i], pixPosRoundTrip[i])

    # def testGetDistortedWcs(self):
    #     """Test utils.getDistortedWcs
    #     """
    #     dw = DetectorWrapper()
    #     detector = dw.detector

    #     # the standard case: the exposure's WCS is pure TAN WCS and distortion information is available;
    #     # return a DistortedTanWcs
    #     exposure = afwImage.ExposureF(10, 10)
    #     exposure.setDetector(detector)
    #     exposure.setWcs(self.tanWcs)
    #     self.assertFalse(self.tanWcs.hasDistortion())
    #     outWcs = getDistortedWcs(exposure.getInfo())
    #     self.assertTrue(outWcs.hasDistortion())
    #     self.assertIsInstance(outWcs, afwImage.DistortedTanWcs)
    #     del exposure  # avoid accidental reuse
    #     del outWcs

    #     # return the original WCS if the exposure's WCS has distortion
    #     pixelsToTanPixels = afwGeom.RadialXYTransform([0, 1.001, 0.00003])
    #     distortedWcs = afwImage.DistortedTanWcs(self.tanWcs, pixelsToTanPixels)
    #     self.assertTrue(distortedWcs.hasDistortion())
    #     exposure = afwImage.ExposureF(10, 10)
    #     exposure.setWcs(distortedWcs)
    #     exposure.setDetector(detector)
    #     outWcs = getDistortedWcs(exposure.getInfo())
    #     self.assertTrue(outWcs.hasDistortion())
    #     self.assertIsInstance(outWcs, afwImage.DistortedTanWcs)
    #     del exposure
    #     del distortedWcs
    #     del outWcs

    #     # raise an exception if exposure has no WCS
    #     exposure = afwImage.ExposureF(10, 10)
    #     exposure.setDetector(detector)
    #     with self.assertRaises(Exception):
    #         getDistortedWcs(exposure.getInfo())
    #     del exposure

    #     # return the original pure TAN WCS if the exposure has no detector
    #     exposure = afwImage.ExposureF(10, 10)
    #     exposure.setWcs(self.tanWcs)
    #     outWcs = getDistortedWcs(exposure.getInfo())
    #     self.assertFalse(outWcs.hasDistortion())
    #     self.assertIsInstance(outWcs, afwImage.TanWcs)
    #     self.assertNotIsInstance(outWcs, afwImage.DistortedTanWcs)
    #     del exposure
    #     del outWcs

    #     # return the original pure TAN WCS if the exposure's detector has no
    #     # TAN_PIXELS transform
    #     def removeTanPixels(detectorWrapper):
    #         tanPixSys = detector.makeCameraSys(TAN_PIXELS)
    #         detectorWrapper.transMap.pop(tanPixSys)
    #     detectorNoTanPix = DetectorWrapper(modFunc=removeTanPixels).detector
    #     exposure = afwImage.ExposureF(10, 10)
    #     exposure.setWcs(self.tanWcs)
    #     exposure.setDetector(detectorNoTanPix)
    #     outWcs = getDistortedWcs(exposure.getInfo())
    #     self.assertFalse(outWcs.hasDistortion())
    #     self.assertIsInstance(outWcs, afwImage.TanWcs)
    #     self.assertNotIsInstance(outWcs, afwImage.DistortedTanWcs)
    #     del exposure
    #     del outWcs


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
