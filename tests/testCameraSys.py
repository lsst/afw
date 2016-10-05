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
#pybind11#Tests for lsst.afw.cameraGeom.CameraSys and CameraSysPrefix
#pybind11#"""
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.cameraGeom as cameraGeom
#pybind11#
#pybind11#
#pybind11#class CameraSysTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def testBasics(self):
#pybind11#        """Test CameraSys and CameraSysPrefix
#pybind11#        """
#pybind11#        for sysName in ("pupil", "pixels"):
#pybind11#            for detectorName in ("", "det1", "det2"):
#pybind11#                cameraSys = cameraGeom.CameraSys(sysName, detectorName)
#pybind11#                self.assertEqual(cameraSys.getSysName(), sysName)
#pybind11#                self.assertEqual(cameraSys.getDetectorName(), detectorName)
#pybind11#                self.assertEqual(cameraSys.hasDetectorName(), bool(detectorName))
#pybind11#
#pybind11#                noDetSys = cameraGeom.CameraSys(sysName)
#pybind11#                self.assertEqual(noDetSys.getSysName(), sysName)
#pybind11#                self.assertEqual(noDetSys.getDetectorName(), "")
#pybind11#                self.assertFalse(noDetSys.hasDetectorName())
#pybind11#
#pybind11#                camSysPrefix = cameraGeom.CameraSysPrefix(sysName)
#pybind11#                self.assertEqual(camSysPrefix.getSysName(), sysName)
#pybind11#
#pybind11#                if detectorName:
#pybind11#                    self.assertNotEqual(cameraSys, noDetSys)
#pybind11#                else:
#pybind11#                    self.assertEqual(cameraSys, noDetSys)
#pybind11#
#pybind11#                # The following tests are checking the functionality of the == and !=
#pybind11#                # operators and should not be replaced with assertEqual ot assertNotEqual
#pybind11#                self.assertTrue(cameraSys != camSysPrefix)
#pybind11#                self.assertTrue(noDetSys != camSysPrefix)
#pybind11#                self.assertFalse(cameraSys == camSysPrefix)
#pybind11#                self.assertFalse(noDetSys == camSysPrefix)
#pybind11#
#pybind11#            for sysName2 in ("pupil", "pixels"):
#pybind11#                for detectorName2 in ("", "det1", "det2"):
#pybind11#                    cameraSys2 = cameraGeom.CameraSys(sysName2, detectorName2)
#pybind11#                    if sysName == sysName2 and detectorName == detectorName2:
#pybind11#                        self.assertEqual(cameraSys, cameraSys2)
#pybind11#                    else:
#pybind11#                        self.assertNotEqual(cameraSys, cameraSys2)
#pybind11#
#pybind11#                    camSysPrefix2 = cameraGeom.CameraSysPrefix(sysName2)
#pybind11#                    if sysName2 == sysName:
#pybind11#                        self.assertEqual(camSysPrefix2, camSysPrefix)
#pybind11#                    else:
#pybind11#                        self.assertNotEqual(camSysPrefix2, camSysPrefix)
#pybind11#
#pybind11#    def testRepr(self):
#pybind11#        """Test __repr__
#pybind11#        """
#pybind11#        cs1 = cameraGeom.CameraSys("pixels", "det1")
#pybind11#        self.assertEqual(repr(cs1), "CameraSys(pixels, det1)")
#pybind11#
#pybind11#        cs2 = cameraGeom.CameraSys("pixels")
#pybind11#        self.assertEqual(repr(cs2), "CameraSys(pixels)")
#pybind11#
#pybind11#        dsp = cameraGeom.CameraSysPrefix("pixels")
#pybind11#        self.assertEqual(repr(dsp), "CameraSysPrefix(pixels)")
#pybind11#
#pybind11#    def testHashing(self):
#pybind11#        """Test that hashing works as expected"""
#pybind11#        cs1 = cameraGeom.CameraSys("pixels", "det1")
#pybind11#        cs1Copy = cameraGeom.CameraSys("pixels", "det1")
#pybind11#        cs2 = cameraGeom.CameraSys("pixels", "det2")
#pybind11#        cs2Copy = cameraGeom.CameraSys("pixels", "det2")
#pybind11#        # import pdb; pdb.set_trace()
#pybind11#        csSet = set((cs1, cs1Copy, cs2, cs2Copy))
#pybind11#        self.assertEqual(len(csSet), 2)
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
