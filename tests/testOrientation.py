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
Tests for lsst.afw.cameraGeom.Orientation

@todo: test the transforms against expected
"""
import unittest

import lsst.utils.tests
import lsst.afw.geom as afwGeom
from lsst.afw.cameraGeom import Orientation

class OrientationWrapper(object):
    def __init__(self,
        fpPosition = afwGeom.Point2D(0, 0),
        refPoint = afwGeom.Point2D(-0.5, -0.5),
        yaw = afwGeom.Angle(0),
        pitch = afwGeom.Angle(0),
        roll = afwGeom.Angle(0),
    ):
        self.fpPosition = fpPosition
        self.refPoint = refPoint
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.orient = Orientation(fpPosition, refPoint, yaw, pitch, roll)

class OrientationTestCase(unittest.TestCase):
    def testDefaultConstructor(self):
        """Test default constructor
        """
        orient = Orientation()
        for i in range(2):
            self.assertAlmostEquals(0, orient.getFpPosition()[i])
            self.assertAlmostEquals(-0.5, orient.getReferencePoint()[i])
        zeroAngle = afwGeom.Angle(0)
        self.assertAlmostEquals(zeroAngle, orient.getYaw())
        self.assertAlmostEquals(zeroAngle, orient.getRoll())
        self.assertAlmostEquals(zeroAngle, orient.getPitch())

        fwdTransform = orient.makeFpPixelTransform(afwGeom.Extent2D(1.0))
        for x in (-100.1, 0.0, 230.0):
            for y in (-45.0, 0.0, 25.1):
                xy = afwGeom.Point2D(x, y)
                fwdXY = fwdTransform.forwardTransform(xy)
                for i in range(2):
                    self.assertAlmostEquals(xy[i] - 0.5, fwdXY[i])
        self.compareTransforms(orient)

    def testGetNQuarter(self):
        """Test the getNQuarter method
        """
        refPos = afwGeom.Point2D(0., 0.)
        fpPos = afwGeom.Point2D(0., 0.)
        angles = ((0., 0), (90., 1), (180., 2), (270., 3), (360., 4),
                  (0.1, 0), (44.9, 0), (45.1, 1), (89.9, 1), (90.1, 1),
                  (134.9, 1), (135.1, 2), (179.9, 2), (180.1, 2), (224.9, 2),
                  (225.1, 3), (269.9, 3), (270.1, 3), (314.9, 3), (315.1, 4),
                  (359.9, 4))
        for angle in angles:
            orient = Orientation(fpPos, refPos, afwGeom.Angle(angle[0], afwGeom.degrees))
            self.assertEquals(orient.getNQuarter(), angle[1])

    def checkTransforms(self, orientWrapper, pixelSize=afwGeom.Extent2D(0.12, 0.21)):
        """Check that the transforms do what we expect them to
        """
        pixToFpTransform = orientWrapper.orient.makeFpPixelTransform(pixelSize)
        for x in (-100.1, 0.0, 230.0):
            for y in (-45.0, 0.0, 25.1):
                pixPos = afwGeom.Point2D(x, y)
                pixToFpTransform.forwardTransform(pixPos)



    def compareTransforms(self, orient, pixelSize=afwGeom.Extent2D(0.12, 0.21)):
        """Compare makeFpPixelTransform and makePixelFpTransform to each other
        """
        fwdTransform = orient.makeFpPixelTransform(pixelSize)
        revTransform = orient.makePixelFpTransform(pixelSize)
        for x in (-100.1, 0.0, 230.0):
            for y in (-45.0, 0.0, 25.1):
                pixPos = afwGeom.Point2D(x, y)
                fwdFPPos  = fwdTransform.forwardTransform(pixPos)
                fwdPixPos = fwdTransform.reverseTransform(fwdFPPos)
                revPixPos = revTransform.forwardTransform(fwdFPPos)
                revFPPos  = revTransform.reverseTransform(pixPos)

                for i in range(2):
                    self.assertAlmostEquals(pixPos[i], fwdPixPos[i])
                    self.assertAlmostEquals(pixPos[i], revPixPos[i])
                    self.assertAlmostEquals(fwdFPPos[i], revFPPos[i])

    def testGetters(self):
        """Test getters
        """
        ow = OrientationWrapper(
            fpPosition = afwGeom.Point2D(0.1, -0.2),
            refPoint = afwGeom.Point2D(-5.7, 42.3),
            yaw = afwGeom.Angle(-0.53),
            pitch = afwGeom.Angle(0.234),
            roll = afwGeom.Angle(1.2),
        )
        for i in range(2):
            self.assertAlmostEquals(ow.fpPosition[i], ow.orient.getFpPosition()[i])
            self.assertAlmostEquals(ow.refPoint[i], ow.orient.getReferencePoint()[i])
        self.assertAlmostEquals(ow.yaw, ow.orient.getYaw())
        self.assertAlmostEquals(ow.roll, ow.orient.getRoll())
        self.assertAlmostEquals(ow.pitch, ow.orient.getPitch())


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(OrientationTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
