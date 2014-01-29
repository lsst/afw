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
"""

import sys
import unittest

import lsst.utils.tests
import lsst.pex.exceptions as pexException
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as cameraGeom


class CameraTransformRegistryTestCase(unittest.TestCase):
    def setUp(self):
        self.nativeSys = cameraGeom.FOCAL_PLANE
        self.radialTransform = afwGeom.RadialXYTransform([0, 0.5, 0.01])
        transMap = {cameraGeom.PUPIL: self.radialTransform}
        self.transReg = cameraGeom.CameraTransformRegistry(self.nativeSys, transMap)

    def tearDown(self):
        self.nativeSys = None
        self.radialTransform = None
        self.transReg = None

    def testBasics(self):
        """Test basic attributes
        """
        self.assertTrue(self.transReg.hasTransform(self.nativeSys))
        self.assertTrue(self.transReg.hasTransform(cameraGeom.PUPIL))
        self.assertFalse(self.transReg.hasTransform(cameraGeom.PIXELS))
        self.assertFalse(self.transReg.hasTransform(cameraGeom.CameraSys("garbage")))

        csList = self.transReg.getCoordSysList()
        self.assertTrue(len(csList) == 2)
        self.assertTrue(self.nativeSys in csList)
        self.assertTrue(cameraGeom.PUPIL in csList)

    def testIteration(self):
        """Test iteration, len and indexing
        """
        self.assertEquals(len(self.transReg), 2)
        trList = []
        for coordSys in self.transReg.getCoordSysList():
            trList.append(self.transReg[coordSys])
        self.assertEquals(len(trList), 2)
        trList2 = [tr for tr in self.transReg]
        self.assertEquals(len(trList2), 2)
        # it would be nice to test equality of the transforms, but it's not worth the fuss


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
