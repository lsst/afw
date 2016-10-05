#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2014 LSST Corporation.
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
#pybind11#"""
#pybind11#Tests for lsst.afw.cameraGeom.Detector
#pybind11#"""
#pybind11#import unittest
#pybind11#
#pybind11#import math
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.cameraGeom as cameraGeom
#pybind11#from lsst.afw.cameraGeom import makePixelToTanPixel
#pybind11#
#pybind11#
#pybind11#class MakePixelToTanPixelTestCaseCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def testSimpleCurvedFocalPlane(self):
#pybind11#        """Test a trivial curved focal plane with square pixels
#pybind11#
#pybind11#        The CCD's lower left pixel is centered on the boresight
#pybind11#        pupil center = focal plane center
#pybind11#        CCD x is along focal plane x
#pybind11#        """
#pybind11#        bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(1000, 1000))
#pybind11#        pixelSizeMm = afwGeom.Extent2D(0.02, 0.02)
#pybind11#        plateScale = 25.0   # arcsec/mm
#pybind11#        yaw = 0 * afwGeom.degrees
#pybind11#        fpPosition = afwGeom.Point2D(0, 0)  # focal-plane position of ref position on detector (mm)
#pybind11#        refPoint = afwGeom.Point2D(0, 0)  # ref position on detector (pos of lower left corner)
#pybind11#        orientation = cameraGeom.Orientation(
#pybind11#            fpPosition,
#pybind11#            refPoint,
#pybind11#            yaw,
#pybind11#        )
#pybind11#        pixelToFocalPlane = orientation.makePixelFpTransform(pixelSizeMm)
#pybind11#        plateScaleRad = afwGeom.Angle(plateScale, afwGeom.arcseconds).asRadians()
#pybind11#        focalPlaneToPupil = afwGeom.RadialXYTransform((0.0, plateScaleRad, 0.0, 0.001 * plateScaleRad))
#pybind11#        pixelToPupil = afwGeom.MultiXYTransform((pixelToFocalPlane, focalPlaneToPupil))
#pybind11#
#pybind11#        pixelToTanPixel = makePixelToTanPixel(
#pybind11#            bbox=bbox,
#pybind11#            orientation=orientation,
#pybind11#            focalPlaneToPupil=focalPlaneToPupil,
#pybind11#            pixelSizeMm=pixelSizeMm,
#pybind11#        )
#pybind11#
#pybind11#        # pupil center should be pixel position 0, 0 and tan pixel position 0, 0
#pybind11#        pixAtPupilCtr = pixelToPupil.reverseTransform(afwGeom.Point2D(0, 0))
#pybind11#        self.assertPairsNearlyEqual(pixAtPupilCtr, [0, 0])
#pybind11#        tanPixAtPupilCr = pixelToTanPixel.forwardTransform(pixAtPupilCtr)
#pybind11#        self.assertPairsNearlyEqual(tanPixAtPupilCr, [0, 0])
#pybind11#
#pybind11#        # build same camera geometry transforms without optical distortion
#pybind11#        focalPlaneToPupilNoDistortion = afwGeom.RadialXYTransform((0.0, plateScaleRad))
#pybind11#        pixelToPupilNoDistortion = afwGeom.MultiXYTransform(
#pybind11#            (pixelToFocalPlane, focalPlaneToPupilNoDistortion))
#pybind11#
#pybind11#        for x in (100, 200, 1000):
#pybind11#            for y in (100, 500, 800):
#pybind11#                pixPos = afwGeom.Point2D(x, y)
#pybind11#                tanPixPos = pixelToTanPixel.forwardTransform(pixPos)
#pybind11#                # pix to tan pix should be radial
#pybind11#                self.assertAlmostEqual(
#pybind11#                    math.atan2(pixPos[1], pixPos[0]),
#pybind11#                    math.atan2(tanPixPos[1], tanPixPos[0]),
#pybind11#                )
#pybind11#
#pybind11#                # for a given pupil angle (which, together with a pointing, gives a position on the sky):
#pybind11#                # - pupil to pixels gives pixPos
#pybind11#                # - undistorted pupil to pixels gives tanPixPos
#pybind11#                pupilPos = pixelToPupil.forwardTransform(pixPos)
#pybind11#                desTanPixPos = pixelToPupilNoDistortion.reverseTransform(pupilPos)
#pybind11#                self.assertPairsNearlyEqual(desTanPixPos, tanPixPos)
#pybind11#
#pybind11#    def testCurvedFocalPlane(self):
#pybind11#        """Test a curved focal plane (with rectangular pixels)
#pybind11#        """
#pybind11#        bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(1000, 1000))
#pybind11#        pixelSizeMm = afwGeom.Extent2D(0.02, 0.03)
#pybind11#        plateScale = 25.0   # arcsec/mm
#pybind11#        yaw = afwGeom.Angle(20, afwGeom.degrees)
#pybind11#        fpPosition = afwGeom.Point2D(50, 25)  # focal-plane position of ref position on detector (mm)
#pybind11#        refPoint = afwGeom.Point2D(-0.5, -0.5)  # ref position on detector (pos of lower left corner)
#pybind11#        orientation = cameraGeom.Orientation(
#pybind11#            fpPosition,
#pybind11#            refPoint,
#pybind11#            yaw,
#pybind11#        )
#pybind11#        pixelToFocalPlane = orientation.makePixelFpTransform(pixelSizeMm)
#pybind11#        plateScaleRad = afwGeom.Angle(plateScale, afwGeom.arcseconds).asRadians()
#pybind11#        focalPlaneToPupil = afwGeom.RadialXYTransform((0.0, plateScaleRad, 0.0, 0.001 * plateScaleRad))
#pybind11#        pixelToPupil = afwGeom.MultiXYTransform((pixelToFocalPlane, focalPlaneToPupil))
#pybind11#
#pybind11#        pixelToTanPixel = makePixelToTanPixel(
#pybind11#            bbox=bbox,
#pybind11#            orientation=orientation,
#pybind11#            focalPlaneToPupil=focalPlaneToPupil,
#pybind11#            pixelSizeMm=pixelSizeMm,
#pybind11#        )
#pybind11#
#pybind11#        # the center point of the pupil frame should not move
#pybind11#        pixAtPupilCtr = pixelToPupil.reverseTransform(afwGeom.Point2D(0, 0))
#pybind11#        tanPixAtPupilCr = pixelToTanPixel.forwardTransform(pixAtPupilCtr)
#pybind11#        self.assertPairsNearlyEqual(pixAtPupilCtr, tanPixAtPupilCr)
#pybind11#
#pybind11#        # build same camera geometry transforms without optical distortion
#pybind11#        focalPlaneToPupilNoDistortion = afwGeom.RadialXYTransform((0.0, plateScaleRad))
#pybind11#        pixelToPupilNoDistortion = afwGeom.MultiXYTransform(
#pybind11#            (pixelToFocalPlane, focalPlaneToPupilNoDistortion))
#pybind11#
#pybind11#        for x in (100, 200, 1000):
#pybind11#            for y in (100, 500, 800):
#pybind11#                pixPos = afwGeom.Point2D(x, y)
#pybind11#                tanPixPos = pixelToTanPixel.forwardTransform(pixPos)
#pybind11#
#pybind11#                # for a given pupil position (which, together with a pointing, gives a position on the sky):
#pybind11#                # - pupil to pixels gives pixPos
#pybind11#                # - undistorted pupil to pixels gives tanPixPos
#pybind11#                pupilPos = pixelToPupil.forwardTransform(pixPos)
#pybind11#                desTanPixPos = pixelToPupilNoDistortion.reverseTransform(pupilPos)
#pybind11#                self.assertPairsNearlyEqual(desTanPixPos, tanPixPos)
#pybind11#
#pybind11#    def testFlatFocalPlane(self):
#pybind11#        """Test an undistorted focal plane (with rectangular pixels)
#pybind11#        """
#pybind11#        bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(1000, 1000))
#pybind11#        pixelSizeMm = afwGeom.Extent2D(0.02, 0.03)
#pybind11#        plateScale = 25.0   # arcsec/mm
#pybind11#        yaw = afwGeom.Angle(20, afwGeom.degrees)
#pybind11#        fpPosition = afwGeom.Point2D(50, 25)  # focal-plane position of ref position on detector (mm)
#pybind11#        refPoint = afwGeom.Point2D(-0.5, -0.5)  # ref position on detector (pos of lower left corner)
#pybind11#        orientation = cameraGeom.Orientation(
#pybind11#            fpPosition,
#pybind11#            refPoint,
#pybind11#            yaw,
#pybind11#        )
#pybind11#        plateScaleRad = afwGeom.Angle(plateScale, afwGeom.arcseconds).asRadians()
#pybind11#        focalPlaneToPupil = afwGeom.RadialXYTransform((0.0, plateScaleRad))
#pybind11#
#pybind11#        pixelToTanPixel = makePixelToTanPixel(
#pybind11#            bbox=bbox,
#pybind11#            orientation=orientation,
#pybind11#            focalPlaneToPupil=focalPlaneToPupil,
#pybind11#            pixelSizeMm=pixelSizeMm,
#pybind11#        )
#pybind11#
#pybind11#        # with no distortion, this should be a unity transform
#pybind11#        for pointPix in (
#pybind11#            afwGeom.Point2D(0, 0),
#pybind11#            afwGeom.Point2D(1000, 2000),
#pybind11#            afwGeom.Point2D(-100.5, 27.23),
#pybind11#        ):
#pybind11#            pointTanPix = pixelToTanPixel.forwardTransform(pointPix)
#pybind11#            self.assertPairsNearlyEqual(pointTanPix, pointPix)
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
