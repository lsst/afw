#!/usr/bin/env python2
from __future__ import absolute_import, division
#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#
"""
Tests for lsst.afw.cameraGeom.Detector
"""
import unittest

import numpy

import lsst.utils.tests
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as cameraGeom
from lsst.afw.cameraGeom import makePixelToTanPixel

class MakePixelToTanPixelTestCaseCase(unittest.TestCase):
    def testCurvedFocalPlane(self):
        """Test a curved focal plane (with rectangular pixels)
        """
        bbox = afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Extent2I(1000, 1000))
        pixelSizeMm = afwGeom.Extent2D(0.02, 0.03)
        plateScale = 25.0   # arcsec/mm
        yaw = afwGeom.Angle(20, afwGeom.degrees)
        fpPosition = afwGeom.Point2D(50, 25) # focal-plane position of ref position on detector (mm)
        refPoint = afwGeom.Point2D(-0.5, -0.5) # ref position on detector (pos of lower left corner)
        orientation = cameraGeom.Orientation(
            fpPosition,
            refPoint,
            yaw,
        )
        plateScaleRad = afwGeom.Angle(plateScale, afwGeom.arcseconds).asRadians()
        focalPlaneToPupil = afwGeom.RadialXYTransform((0.0, plateScaleRad, 0.0, 0.001 * plateScaleRad))

        pixelToTanPixel = makePixelToTanPixel(
            bbox = bbox,
            orientation = orientation,
            focalPlaneToPupil = focalPlaneToPupil,
            pixelSizeMm = pixelSizeMm,
            plateScale = plateScale,
        )

        # the center point of the detector should not move
        ctrPointPix = afwGeom.Box2D(bbox).getCenter()
        ctrPointTanPix = pixelToTanPixel.forwardTransform(ctrPointPix)
        for i in range(2):
            self.assertAlmostEquals(ctrPointTanPix[i], ctrPointPix[i])

        # two points separated by x pixels in tan pixels coordinates
        # should be separated x * rad/tanPix in pupil coordinates,
        # where rad/tanPix = plate scale in rad/MM * mean pixel size in mm
        radPerTanPixel = plateScaleRad * (pixelSizeMm[0] + pixelSizeMm[1]) / 2.0
        pixelToFocalPlane = orientation.makePixelFpTransform(pixelSizeMm)
        pixelToPupil = afwGeom.MultiXYTransform((pixelToFocalPlane, focalPlaneToPupil))
        prevPointPupil = None
        prevPointTanPix = None
        for pointPix in (
            afwGeom.Point2D(0, 0),
            afwGeom.Point2D(1000, 2000),
            afwGeom.Point2D(-100.5, 27.23),
            afwGeom.Point2D(-95.3, 0.0),
        ):
            pointPupil = pixelToPupil.forwardTransform(pointPix)
            pointTanPix = pixelToTanPixel.forwardTransform(pointPix)
            if prevPointPupil:
                pupilSep = numpy.linalg.norm(pointPupil - prevPointPupil)
                tanPixSep = numpy.linalg.norm(pointTanPix - prevPointTanPix)
                self.assertAlmostEquals(tanPixSep * radPerTanPixel, pupilSep)
            prevPointPupil = pointPupil
            prevPointTanPix = pointTanPix


    def testFlatFocalPlane(self):
        """Test an undistorted focal plane (with rectangular pixels)
        """
        bbox = afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Extent2I(1000, 1000))
        pixSizeFactor = numpy.array((1.2, 0.8))
        pixelSizeMm = afwGeom.Extent2D(0.02 * pixSizeFactor[0], 0.02 * pixSizeFactor[1])
        plateScale = 25.0   # arcsec/mm
        yaw = afwGeom.Angle(20, afwGeom.degrees)
        fpPosition = afwGeom.Point2D(50, 25) # focal-plane position of ref position on detector (mm)
        refPoint = afwGeom.Point2D(-0.5, -0.5) # ref position on detector (pos of lower left corner)
        orientation = cameraGeom.Orientation(
            fpPosition,
            refPoint,
            yaw,
        )
        plateScaleRad = afwGeom.Angle(plateScale, afwGeom.arcseconds).asRadians()
        focalPlaneToPupil = afwGeom.RadialXYTransform((0.0, plateScaleRad))

        pixelToTanPixel = makePixelToTanPixel(
            bbox = bbox,
            orientation = orientation,
            focalPlaneToPupil = focalPlaneToPupil,
            pixelSizeMm = pixelSizeMm,
            plateScale = plateScale,
        )

        # with no distortion, this should be a unity transform
        ctrPointPix = numpy.array(afwGeom.Box2D(bbox).getCenter())
        for pointPix in (
            afwGeom.Point2D(0, 0),
            afwGeom.Point2D(1000, 2000),
            afwGeom.Point2D(-100.5, 27.23),
        ):
            pointTanPix = pixelToTanPixel.forwardTransform(pointPix)
            predPointTanPix = ((numpy.array(pointPix) - ctrPointPix) * pixSizeFactor) + ctrPointPix
            for i in range(2):
                self.assertAlmostEquals(pointTanPix[i], predPointTanPix[i])

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(MakePixelToTanPixelTestCaseCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
