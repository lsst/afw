#!/usr/bin/env python2
from __future__ import absolute_import, division
#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#
"""
Tests for lsst.afw.cameraGeom.CameraSys and CameraSysPrefix
"""
import unittest

import lsst.utils.tests
import lsst.afw.cameraGeom as cameraGeom

class CameraSysTestCase(unittest.TestCase):
    def testBasics(self):
        """Test CameraSys and CameraSysPrefix
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

                camSysPrefix = cameraGeom.CameraSysPrefix(sysName)
                self.assertEquals(camSysPrefix.getSysName(), sysName)

                if detectorName:
                    self.assertFalse(cameraSys == noDetSys)
                    self.assertTrue(cameraSys != noDetSys)
                else:
                    self.assertTrue(cameraSys == noDetSys)
                    self.assertFalse(cameraSys != noDetSys)

                self.assertTrue(cameraSys != camSysPrefix)
                self.assertTrue(noDetSys != camSysPrefix)
                self.assertFalse(cameraSys == camSysPrefix)
                self.assertFalse(noDetSys == camSysPrefix)

            for sysName2 in ("pupil", "pixels"):
                for detectorName2 in ("", "det1", "det2"):
                    cameraSys2 = cameraGeom.CameraSys(sysName2, detectorName2)
                    if sysName == sysName2 and detectorName == detectorName2:
                        self.assertTrue(cameraSys == cameraSys2)
                        self.assertFalse(cameraSys != cameraSys2)
                    else:
                        self.assertFalse(cameraSys == cameraSys2)
                        self.assertTrue(cameraSys != cameraSys2)

                    camSysPrefix2 = cameraGeom.CameraSysPrefix(sysName2)
                    if sysName2 == sysName:
                        self.assertTrue(camSysPrefix2 == camSysPrefix)
                        self.assertFalse(camSysPrefix2 != camSysPrefix)
                    else:
                        self.assertFalse(camSysPrefix2 == camSysPrefix)
                        self.assertTrue(camSysPrefix2 != camSysPrefix)

    def testRepr(self):
        """Test __repr__
        """
        cs1 = cameraGeom.CameraSys("pixels", "det1")
        self.assertEqual(repr(cs1), "CameraSys(pixels, det1)")

        cs2 = cameraGeom.CameraSys("pixels")
        self.assertEqual(repr(cs2), "CameraSys(pixels)")

        dsp = cameraGeom.CameraSysPrefix("pixels")
        self.assertEqual(repr(dsp), "CameraSysPrefix(pixels)")

    def testHashing(self):
        """Test that hashing works as expected"""
        cs1 = cameraGeom.CameraSys("pixels", "det1")
        cs1Copy = cameraGeom.CameraSys("pixels", "det1")
        cs2 = cameraGeom.CameraSys("pixels", "det2")
        cs2Copy = cameraGeom.CameraSys("pixels", "det2")
        # import pdb; pdb.set_trace()
        csSet = set((cs1, cs1Copy, cs2, cs2Copy))
        self.assertEquals(len(csSet), 2)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(CameraSysTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
