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
Tests for lsst.afw.cameraGeom.CameraSys and CameraSysPrefix
"""
from __future__ import absolute_import, division, print_function
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
                self.assertEqual(cameraSys.getSysName(), sysName)
                self.assertEqual(cameraSys.getDetectorName(), detectorName)
                self.assertEqual(cameraSys.hasDetectorName(),
                                 bool(detectorName))

                noDetSys = cameraGeom.CameraSys(sysName)
                self.assertEqual(noDetSys.getSysName(), sysName)
                self.assertEqual(noDetSys.getDetectorName(), "")
                self.assertFalse(noDetSys.hasDetectorName())

                camSysPrefix = cameraGeom.CameraSysPrefix(sysName)
                self.assertEqual(camSysPrefix.getSysName(), sysName)

                cameraSys2 = cameraGeom.CameraSys(camSysPrefix, detectorName)
                self.assertEqual(cameraSys2.getSysName(), sysName)
                self.assertEqual(cameraSys2.getDetectorName(), detectorName)
                self.assertEqual(cameraSys2, cameraSys)

                if detectorName:
                    self.assertNotEqual(cameraSys, noDetSys)
                else:
                    self.assertEqual(cameraSys, noDetSys)

                # The following tests are checking the functionality of the == and !=
                # operators and should not be replaced with assertEqual ot
                # assertNotEqual
                self.assertTrue(cameraSys != camSysPrefix)
                self.assertTrue(noDetSys != camSysPrefix)
                self.assertFalse(cameraSys == camSysPrefix)
                self.assertFalse(noDetSys == camSysPrefix)

            for sysName2 in ("pupil", "pixels"):
                for detectorName2 in ("", "det1", "det2"):
                    cameraSys2 = cameraGeom.CameraSys(sysName2, detectorName2)
                    if sysName == sysName2 and detectorName == detectorName2:
                        self.assertEqual(cameraSys, cameraSys2)
                    else:
                        self.assertNotEqual(cameraSys, cameraSys2)

                    camSysPrefix2 = cameraGeom.CameraSysPrefix(sysName2)
                    if sysName2 == sysName:
                        self.assertEqual(camSysPrefix2, camSysPrefix)
                    else:
                        self.assertNotEqual(camSysPrefix2, camSysPrefix)

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
        self.assertEqual(len(csSet), 2)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
