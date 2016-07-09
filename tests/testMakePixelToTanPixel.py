#!/usr/bin/env python2
from __future__ import absolute_import, division
# 
# LSST Data Management System
# Copyright 2014 LSST Corporation.
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
Tests for lsst.afw.cameraGeom.Detector
"""
import unittest

import math

import lsst.utils.tests
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as cameraGeom
from lsst.afw.cameraGeom import makePixelToTanPixel

class MakePixelToTanPixelTestCaseCase(lsst.utils.tests.TestCase):
    def testSimpleCurvedFocalPlane(self):
        """Test a trivial curved focal plane with square pixels

        The CCD's lower left pixel is centered on the boresight
        pupil center = focal plane center
        CCD x is along focal plane x
        """
        bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(1000, 1000))
        pixelSizeMm = afwGeom.Extent2D(0.02, 0.02)
        plateScale = 25.0   # arcsec/mm
        yaw = 0 * afwGeom.degrees
        fpPosition = afwGeom.Point2D(0, 0)  # focal-plane position of ref position on detector (mm)
        refPoint = afwGeom.Point2D(0, 0)  # ref position on detector (pos of lower left corner)
        orientation = cameraGeom.Orientation(
            fpPosition,
            refPoint,
            yaw,
        )
        pixelToFocalPlane = orientation.makePixelFpTransform(pixelSizeMm)
        plateScaleRad = afwGeom.Angle(plateScale, afwGeom.arcseconds).asRadians()
        focalPlaneToPupil = afwGeom.RadialXYTransform((0.0, plateScaleRad, 0.0, 0.001 * plateScaleRad))
        pixelToPupil = afwGeom.MultiXYTransform((pixelToFocalPlane, focalPlaneToPupil))

        pixelToTanPixel = makePixelToTanPixel(
            bbox=bbox,
            orientation=orientation,
            focalPlaneToPupil=focalPlaneToPupil,
            pixelSizeMm=pixelSizeMm,
            plateScale=plateScale,
        )

        # pupil center should be pixel position 0, 0 and tan pixel position 0, 0
        pixAtPupilCtr = pixelToPupil.reverseTransform(afwGeom.Point2D(0, 0))
        self.assertPairsNearlyEqual(pixAtPupilCtr, [0, 0])
        tanPixAtPupilCr = pixelToTanPixel.forwardTransform(pixAtPupilCtr)
        self.assertPairsNearlyEqual(tanPixAtPupilCr, [0, 0])

        # build same camera geometry transforms without optical distortion
        focalPlaneToPupilNoDistortion = afwGeom.RadialXYTransform((0.0, plateScaleRad))
        pixelToPupilNoDistortion = afwGeom.MultiXYTransform(
            (pixelToFocalPlane, focalPlaneToPupilNoDistortion))

        for x in (100, 200, 1000):
            for y in (100, 500, 800):
                pixPos = afwGeom.Point2D(x, y)
                tanPixPos = pixelToTanPixel.forwardTransform(pixPos)
                # pix to tan pix should be radial
                self.assertAlmostEqual(
                    math.atan2(pixPos[1], pixPos[0]),
                    math.atan2(tanPixPos[1], tanPixPos[0]),
                )

                # for a given pupil angle (which, together with a pointing, gives a position on the sky):
                # - pupil to pixels gives pixPos
                # - undistorted pupil to pixels gives tanPixPos
                pupilPos = pixelToPupil.forwardTransform(pixPos)
                desTanPixPos = pixelToPupilNoDistortion.reverseTransform(pupilPos)
                self.assertPairsNearlyEqual(desTanPixPos, tanPixPos)

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
        pixelToFocalPlane = orientation.makePixelFpTransform(pixelSizeMm)
        plateScaleRad = afwGeom.Angle(plateScale, afwGeom.arcseconds).asRadians()
        focalPlaneToPupil = afwGeom.RadialXYTransform((0.0, plateScaleRad, 0.0, 0.001 * plateScaleRad))
        pixelToPupil = afwGeom.MultiXYTransform((pixelToFocalPlane, focalPlaneToPupil))

        pixelToTanPixel = makePixelToTanPixel(
            bbox = bbox,
            orientation = orientation,
            focalPlaneToPupil = focalPlaneToPupil,
            pixelSizeMm = pixelSizeMm,
            plateScale = plateScale,
        )

        # the center point of the pupil frame should not move
        pixAtPupilCtr = pixelToPupil.reverseTransform(afwGeom.Point2D(0, 0))
        tanPixAtPupilCr = pixelToTanPixel.forwardTransform(pixAtPupilCtr)
        for i in range(2):
            self.assertAlmostEquals(pixAtPupilCtr[i], tanPixAtPupilCr[i])

        # build same camera geometry transforms without optical distortion
        focalPlaneToPupilNoDistortion = afwGeom.RadialXYTransform((0.0, plateScaleRad))
        pixelToPupilNoDistortion = afwGeom.MultiXYTransform(
            (pixelToFocalPlane, focalPlaneToPupilNoDistortion))

        for x in (100, 200, 1000):
            for y in (100, 500, 800):
                pixPos = afwGeom.Point2D(x, y)
                tanPixPos = pixelToTanPixel.forwardTransform(pixPos)

                # for a given pupil position (which, together with a pointing, gives a position on the sky):
                # - pupil to pixels gives pixPos
                # - undistorted pupil to pixels gives tanPixPos
                pupilPos = pixelToPupil.forwardTransform(pixPos)
                desTanPixPos = pixelToPupilNoDistortion.reverseTransform(pupilPos)
                self.assertPairsNearlyEqual(desTanPixPos, tanPixPos)


    def testFlatFocalPlane(self):
        """Test an undistorted focal plane (with rectangular pixels)
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
        focalPlaneToPupil = afwGeom.RadialXYTransform((0.0, plateScaleRad))

        pixelToTanPixel = makePixelToTanPixel(
            bbox = bbox,
            orientation = orientation,
            focalPlaneToPupil = focalPlaneToPupil,
            pixelSizeMm = pixelSizeMm,
            plateScale = plateScale,
        )

        # with no distortion, this should be a unity transform
        for pointPix in (
            afwGeom.Point2D(0, 0),
            afwGeom.Point2D(1000, 2000),
            afwGeom.Point2D(-100.5, 27.23),
        ):
            pointTanPix = pixelToTanPixel.forwardTransform(pointPix)
            for i in range(2):
                self.assertAlmostEquals(pointTanPix[i], pointPix[i])

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
