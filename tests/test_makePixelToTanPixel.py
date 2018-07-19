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
import lsst.geom
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
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(1000, 1000), invert=False)
        pixelSizeMm = lsst.geom.Extent2D(0.02, 0.02)
        plateScale = 25.0   # arcsec/mm
        yaw = 0 * lsst.geom.degrees
        # focal-plane position of ref position on detector (mm)
        fpPosition = lsst.geom.Point2D(0, 0)
        # ref position on detector (pos of lower left corner)
        refPoint = lsst.geom.Point2D(0, 0)
        orientation = cameraGeom.Orientation(
            fpPosition,
            refPoint,
            yaw,
        )
        pixelToFocalPlane = orientation.makePixelFpTransform(pixelSizeMm)
        plateScaleRad = lsst.geom.Angle(  # rad/mm
            plateScale, lsst.geom.arcseconds).asRadians()
        focalPlaneToField = afwGeom.makeRadialTransform(
            (0.0, plateScaleRad, 0.0, 0.001 * plateScaleRad))
        pixelToField = pixelToFocalPlane.then(focalPlaneToField)

        pixelToTanPixel = makePixelToTanPixel(
            bbox=bbox,
            orientation=orientation,
            focalPlaneToField=focalPlaneToField,
            pixelSizeMm=pixelSizeMm,
        )

        # field center should be pixel position 0, 0 and tan pixel position 0,
        # 0
        pixAtFieldCtr = pixelToField.applyInverse(lsst.geom.Point2D(0, 0))
        self.assertPairsAlmostEqual(pixAtFieldCtr, [0, 0])
        tanPixAtFieldCr = pixelToTanPixel.applyForward(pixAtFieldCtr)
        self.assertPairsAlmostEqual(tanPixAtFieldCr, [0, 0])

        # build same camera geometry transforms without optical distortion
        focalPlaneToFieldNoDistortion = afwGeom.makeRadialTransform(
            (0.0, plateScaleRad))
        pixelToFieldNoDistortion = pixelToFocalPlane.then(focalPlaneToFieldNoDistortion)

        for x in (100, 200, 1000):
            for y in (100, 500, 800):
                pixPos = lsst.geom.Point2D(x, y)
                tanPixPos = pixelToTanPixel.applyForward(pixPos)
                # pix to tan pix should be radial
                self.assertAlmostEqual(
                    math.atan2(pixPos[1], pixPos[0]),
                    math.atan2(tanPixPos[1], tanPixPos[0]),
                )

                # for a given field angle (which, together with a pointing, gives a position on the sky):
                # - field angle to pixels gives pixPos
                # - undistorted field anle to pixels gives tanPixPos
                fieldPos = pixelToField.applyForward(pixPos)
                desTanPixPos = pixelToFieldNoDistortion.applyInverse(
                    fieldPos)
                self.assertPairsAlmostEqual(desTanPixPos, tanPixPos)

    def testCurvedFocalPlane(self):
        """Test a curved focal plane (with rectangular pixels)
        """
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(1000, 1000), invert=False)
        pixelSizeMm = lsst.geom.Extent2D(0.02, 0.03)
        plateScale = 25.0   # arcsec/mm
        yaw = lsst.geom.Angle(20, lsst.geom.degrees)
        # focal-plane position of ref position on detector (mm)
        fpPosition = lsst.geom.Point2D(50, 25)
        # ref position on detector (pos of lower left corner)
        refPoint = lsst.geom.Point2D(-0.5, -0.5)
        orientation = cameraGeom.Orientation(
            fpPosition,
            refPoint,
            yaw,
        )
        pixelToFocalPlane = orientation.makePixelFpTransform(pixelSizeMm)
        plateScaleRad = lsst.geom.Angle(
            plateScale, lsst.geom.arcseconds).asRadians()
        focalPlaneToField = afwGeom.makeRadialTransform(
            (0.0, plateScaleRad, 0.0, 0.001 * plateScaleRad))
        pixelToField = pixelToFocalPlane.then(focalPlaneToField)

        pixelToTanPixel = makePixelToTanPixel(
            bbox=bbox,
            orientation=orientation,
            focalPlaneToField=focalPlaneToField,
            pixelSizeMm=pixelSizeMm,
        )

        # the center point of the field angle frame should not move
        pixAtFieldCtr = pixelToField.applyInverse(lsst.geom.Point2D(0, 0))
        tanPixAtFieldCr = pixelToTanPixel.applyForward(pixAtFieldCtr)
        self.assertPairsAlmostEqual(pixAtFieldCtr, tanPixAtFieldCr)

        # build same camera geometry transforms without optical distortion
        focalPlaneToFieldNoDistortion = afwGeom.makeRadialTransform(
            (0.0, plateScaleRad))
        pixelToFieldNoDistortion = pixelToFocalPlane.then(focalPlaneToFieldNoDistortion)

        for x in (100, 200, 1000):
            for y in (100, 500, 800):
                pixPos = lsst.geom.Point2D(x, y)
                tanPixPos = pixelToTanPixel.applyForward(pixPos)

                # for a given field angle (which, together with a pointing, gives a position on the sky):
                # - field angle to pixels gives pixPos
                # - undistorted field angle to pixels gives tanPixPos
                fieldPos = pixelToField.applyForward(pixPos)
                desTanPixPos = pixelToFieldNoDistortion.applyInverse(
                    fieldPos)
                # use a degraded accuracy because small Jacobian errors accumulate this far from the center
                self.assertPairsAlmostEqual(desTanPixPos, tanPixPos, maxDiff=1e-5)

    def testFlatFocalPlane(self):
        """Test an undistorted focal plane (with rectangular pixels)
        """
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(1000, 1000), invert=False)
        pixelSizeMm = lsst.geom.Extent2D(0.02, 0.03)
        plateScale = 25.0   # arcsec/mm
        yaw = lsst.geom.Angle(20, lsst.geom.degrees)
        # focal-plane position of ref position on detector (mm)
        fpPosition = lsst.geom.Point2D(50, 25)
        # ref position on detector (pos of lower left corner)
        refPoint = lsst.geom.Point2D(-0.5, -0.5)
        orientation = cameraGeom.Orientation(
            fpPosition,
            refPoint,
            yaw,
        )
        plateScaleRad = lsst.geom.Angle(
            plateScale, lsst.geom.arcseconds).asRadians()
        focalPlaneToField = afwGeom.makeRadialTransform((0.0, plateScaleRad))

        pixelToTanPixel = makePixelToTanPixel(
            bbox=bbox,
            orientation=orientation,
            focalPlaneToField=focalPlaneToField,
            pixelSizeMm=pixelSizeMm,
        )

        # with no distortion, this should be a unity transform
        for pointPix in (
            lsst.geom.Point2D(0, 0),
            lsst.geom.Point2D(1000, 2000),
            lsst.geom.Point2D(-100.5, 27.23),
        ):
            pointTanPix = pixelToTanPixel.applyForward(pointPix)
            self.assertPairsAlmostEqual(pointTanPix, pointPix)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
