#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from builtins import range
#pybind11#from builtins import object
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
#pybind11#Tests for lsst.afw.cameraGeom.Orientation
#pybind11#
#pybind11#@todo: test the transforms against expected
#pybind11#"""
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#from lsst.afw.cameraGeom import Orientation
#pybind11#
#pybind11#
#pybind11#class OrientationWrapper(object):
#pybind11#
#pybind11#    def __init__(self,
#pybind11#                 fpPosition=afwGeom.Point2D(0, 0),
#pybind11#                 refPoint=afwGeom.Point2D(-0.5, -0.5),
#pybind11#                 yaw=afwGeom.Angle(0),
#pybind11#                 pitch=afwGeom.Angle(0),
#pybind11#                 roll=afwGeom.Angle(0),
#pybind11#                 ):
#pybind11#        self.fpPosition = fpPosition
#pybind11#        self.refPoint = refPoint
#pybind11#        self.yaw = yaw
#pybind11#        self.pitch = pitch
#pybind11#        self.roll = roll
#pybind11#        self.orient = Orientation(fpPosition, refPoint, yaw, pitch, roll)
#pybind11#
#pybind11#
#pybind11#class OrientationTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def testDefaultConstructor(self):
#pybind11#        """Test default constructor
#pybind11#        """
#pybind11#        orient = Orientation()
#pybind11#        for i in range(2):
#pybind11#            self.assertAlmostEqual(0, orient.getFpPosition()[i])
#pybind11#            self.assertAlmostEqual(-0.5, orient.getReferencePoint()[i])
#pybind11#        zeroAngle = afwGeom.Angle(0)
#pybind11#        self.assertAlmostEqual(zeroAngle, orient.getYaw())
#pybind11#        self.assertAlmostEqual(zeroAngle, orient.getRoll())
#pybind11#        self.assertAlmostEqual(zeroAngle, orient.getPitch())
#pybind11#
#pybind11#        fwdTransform = orient.makeFpPixelTransform(afwGeom.Extent2D(1.0))
#pybind11#        for x in (-100.1, 0.0, 230.0):
#pybind11#            for y in (-45.0, 0.0, 25.1):
#pybind11#                xy = afwGeom.Point2D(x, y)
#pybind11#                fwdXY = fwdTransform.forwardTransform(xy)
#pybind11#                for i in range(2):
#pybind11#                    self.assertAlmostEqual(xy[i] - 0.5, fwdXY[i])
#pybind11#        self.compareTransforms(orient)
#pybind11#
#pybind11#    def testGetNQuarter(self):
#pybind11#        """Test the getNQuarter method
#pybind11#        """
#pybind11#        refPos = afwGeom.Point2D(0., 0.)
#pybind11#        fpPos = afwGeom.Point2D(0., 0.)
#pybind11#        angles = ((0., 0), (90., 1), (180., 2), (270., 3), (360., 4),
#pybind11#                  (0.1, 0), (44.9, 0), (45.1, 1), (89.9, 1), (90.1, 1),
#pybind11#                  (134.9, 1), (135.1, 2), (179.9, 2), (180.1, 2), (224.9, 2),
#pybind11#                  (225.1, 3), (269.9, 3), (270.1, 3), (314.9, 3), (315.1, 4),
#pybind11#                  (359.9, 4))
#pybind11#        for angle in angles:
#pybind11#            orient = Orientation(fpPos, refPos, afwGeom.Angle(angle[0], afwGeom.degrees))
#pybind11#            self.assertEqual(orient.getNQuarter(), angle[1])
#pybind11#
#pybind11#    def checkTransforms(self, orientWrapper, pixelSize=afwGeom.Extent2D(0.12, 0.21)):
#pybind11#        """Check that the transforms do what we expect them to
#pybind11#        """
#pybind11#        pixToFpTransform = orientWrapper.orient.makeFpPixelTransform(pixelSize)
#pybind11#        for x in (-100.1, 0.0, 230.0):
#pybind11#            for y in (-45.0, 0.0, 25.1):
#pybind11#                pixPos = afwGeom.Point2D(x, y)
#pybind11#                pixToFpTransform.forwardTransform(pixPos)
#pybind11#
#pybind11#    def compareTransforms(self, orient, pixelSize=afwGeom.Extent2D(0.12, 0.21)):
#pybind11#        """Compare makeFpPixelTransform and makePixelFpTransform to each other
#pybind11#        """
#pybind11#        fwdTransform = orient.makeFpPixelTransform(pixelSize)
#pybind11#        revTransform = orient.makePixelFpTransform(pixelSize)
#pybind11#        for x in (-100.1, 0.0, 230.0):
#pybind11#            for y in (-45.0, 0.0, 25.1):
#pybind11#                pixPos = afwGeom.Point2D(x, y)
#pybind11#                fwdFPPos = fwdTransform.forwardTransform(pixPos)
#pybind11#                fwdPixPos = fwdTransform.reverseTransform(fwdFPPos)
#pybind11#                revPixPos = revTransform.forwardTransform(fwdFPPos)
#pybind11#                revFPPos = revTransform.reverseTransform(pixPos)
#pybind11#
#pybind11#                for i in range(2):
#pybind11#                    self.assertAlmostEqual(pixPos[i], fwdPixPos[i])
#pybind11#                    self.assertAlmostEqual(pixPos[i], revPixPos[i])
#pybind11#                    self.assertAlmostEqual(fwdFPPos[i], revFPPos[i])
#pybind11#
#pybind11#    def testGetters(self):
#pybind11#        """Test getters
#pybind11#        """
#pybind11#        ow = OrientationWrapper(
#pybind11#            fpPosition=afwGeom.Point2D(0.1, -0.2),
#pybind11#            refPoint=afwGeom.Point2D(-5.7, 42.3),
#pybind11#            yaw=afwGeom.Angle(-0.53),
#pybind11#            pitch=afwGeom.Angle(0.234),
#pybind11#            roll=afwGeom.Angle(1.2),
#pybind11#        )
#pybind11#        for i in range(2):
#pybind11#            self.assertAlmostEqual(ow.fpPosition[i], ow.orient.getFpPosition()[i])
#pybind11#            self.assertAlmostEqual(ow.refPoint[i], ow.orient.getReferencePoint()[i])
#pybind11#        self.assertAlmostEqual(ow.yaw, ow.orient.getYaw())
#pybind11#        self.assertAlmostEqual(ow.roll, ow.orient.getRoll())
#pybind11#        self.assertAlmostEqual(ow.pitch, ow.orient.getPitch())
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
