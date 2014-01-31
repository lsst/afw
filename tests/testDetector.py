#!/usr/bin/env python
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

import lsst.utils.tests
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as cameraGeom


class DetectorTestCase(unittest.TestCase):
    def testConstructor(self):
        """Test constructor
        """
        ampList = []
        for name in ("amp 1", "amp 2", "amp 3"):
            bbox = afwGeom.Box2I(afwGeom.Point2I(-1, 1), afwGeom.Extent2I(5, 6))
            gain = 1.71234e3
            readNoise = 0.521237e2
            rawAmplifier = cameraGeom.RawAmplifier(
                afwGeom.Box2I(afwGeom.Point2I(-25, 2), afwGeom.Extent2I(550, 629)),
                afwGeom.Box2I(afwGeom.Point2I(-2, 29), afwGeom.Extent2I(123, 307)),
                afwGeom.Box2I(afwGeom.Point2I(150, 29), afwGeom.Extent2I(25, 307)),
                afwGeom.Box2I(afwGeom.Point2I(-2, 201), afwGeom.Extent2I(123, 6)),
                afwGeom.Box2I(afwGeom.Point2I(-20, 2), afwGeom.Extent2I(5, 307)),
                False,
                True,
                afwGeom.Extent2I(-97, 253),
            )
            ampList.append(cameraGeom.Amplifier(name, bbox, gain, readNoise, rawAmplifier))

        pixelSize = 0.02

        focalPlaneTransform = afwGeom.RadialXYTransform([0, pixelSize])
        distortionTransform = afwGeom.RadialXYTransform([0, 0.95, 0.01])
        transMap = {
            cameraGeom.CameraSys(cameraGeom.PIXELS, name): focalPlaneTransform,
            cameraGeom.CameraSys(cameraGeom.ACTUAL_PIXELS, name): distortionTransform,
        }
        detector = cameraGeom.Detector(
            "detector 1",
            cameraGeom.SCIENCE,
            "xkcd722",
            ampList,
            cameraGeom.Orientation(),
            pixelSize,
            transMap,
        )


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(DetectorTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
