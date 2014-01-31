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
    def setUp(self):
        self.name = "detector 1"
        self.type = cameraGeom.SCIENCE
        self.serial = "xkcd722"
        self.ampList = []
        for name in ("amp 1", "amp 2", "amp 3"):
            bbox = afwGeom.Box2I(afwGeom.Point2I(-1, 1), afwGeom.Extent2I(5, 6))
            gain = 1.71234e3
            readNoise = 0.521237e2
            self.ampList.append(cameraGeom.Amplifier(name, bbox, gain, readNoise, None))
        self.oriention = cameraGeom.Orientation()
        self.pixelSize = 0.02
        self.transMap = {
            cameraGeom.CameraSys(cameraGeom.PIXELS, name): afwGeom.RadialXYTransform([0, self.pixelSize]),
            cameraGeom.CameraSys(cameraGeom.ACTUAL_PIXELS, name): afwGeom.RadialXYTransform([0, 0.95, 0.01]),
        }
        self.detector = cameraGeom.Detector(
            self.name,
            self.type,
            self.serial,
            self.ampList,
            self.oriention,
            self.pixelSize,
            self.transMap,
        )

    def tearDown(self):
        self.ampList = None
        self.pixelSize = None
        self.transMap = None
        self.detector = None

    def testConstructor(self):
        """Test constructor
        """
        detector = self.detector
        self.assertEquals(self.name,   detector.getName())
        self.assertEquals(self.type,   detector.getType())
        self.assertEquals(self.serial, detector.getSerial())
        self.assertAlmostEquals(self.pixelSize, detector.getPixelSize())
        self.assertEquals(len(detector), 3)


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
