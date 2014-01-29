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
Tests for lsst.afw.cameraGeom.CameraTransformRegistry
as well as some basic tests for CameraSys (which is its key)
"""

import sys
import unittest

import lsst.utils.tests
import lsst.pex.exceptions as pexException
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as cameraGeom


class CameraTransformRegistryTestCase(unittest.TestCase):
    def testCameraSys(self):
        """Test CameraSys and DetectorSysPrefix
        """
        for sysName in ("pupil", "pixels"):
            for detectorName in ("", "det1", "det2"):
                cameraSys = cameraGeom.CameraSys(sysName, detectorName)
                self.assertEquals(cameraSys.getSysName(), sysName)
                self.assertEquals(cameraSys.getDetectorName(), detectorName)
                self.assertEquals(cameraSys.hasDetectorName(), bool(detectorName))

                noDetSys = cameraGeom.CameraSys(sysName)
                self.assertEquals(noDetSys.getSysName(), sysName)
                self.assertEquals(noDetSys.getDetectorName(), "")
                self.assertFalse(noDetSys.hasDetectorName())

                detSysPrefix = cameraGeom.DetectorSysPrefix(sysName)
                self.assertEquals(detSysPrefix.getSysName(), sysName)
                self.assertEquals(detSysPrefix.getDetectorName(), "")
                self.assertFalse(detSysPrefix.hasDetectorName())
                self.assertTrue(noDetSys == detSysPrefix)
                self.assertFalse(noDetSys != detSysPrefix)

                if detectorName:
                    self.assertFalse(cameraSys == noDetSys)
                    self.assertTrue(cameraSys != noDetSys)
                else:
                    self.assertTrue(cameraSys == noDetSys)
                    self.assertFalse(cameraSys != noDetSys)

            for sysName2 in ("pupil", "pixels"):
                for detectorName2 in ("", "det1", "det2"):
                    cameraSys2 = cameraGeom.CameraSys(sysName2, detectorName2)
                    if sysName == sysName2 and detectorName == detectorName2:
                        self.assertTrue(cameraSys == cameraSys2)
                        self.assertFalse(cameraSys != cameraSys2)
                    else:
                        self.assertFalse(cameraSys == cameraSys2)
                        self.assertTrue(cameraSys != cameraSys2)

                    detSysPrefix2 = cameraGeom.DetectorSysPrefix(sysName2)
                    if sysName2 == sysName:
                        self.assertTrue(detSysPrefix2 == detSysPrefix)
                        self.assertFalse(detSysPrefix2 != detSysPrefix)
                    else:
                        self.assertFalse(detSysPrefix2 == detSysPrefix)
                        self.assertTrue(detSysPrefix2 != detSysPrefix)

    def testSimpleTransforms(self):
        """Test a simple CameraTransformRegistry
        """
        nativeSys = cameraGeom.FOCAL_PLANE
        radialTransform = afwGeom.RadialXYTransform([0, 0.5, 0.01])
        transList = [(cameraGeom.PUPIL, radialTransform)]
        transReg = cameraGeom.CameraTransformRegistry(nativeSys, transList)

        self.assertTrue(transReg.hasTransform(nativeSys))
        self.assertTrue(transReg.hasTransform(cameraGeom.PUPIL))
        self.assertFalse(transReg.hasTransform(cameraGeom.PIXELS))

        csList = transReg.getCoordSysList()
        self.assertTrue(len(csList) == 2)
        self.assertTrue(nativeSys in csList)
        self.assertTrue(cameraGeom.PUPIL in csList)

        print transReg.getTransformList()
        # self.assertTrue(len(trList) == 2)



#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(CameraTransformRegistryTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
