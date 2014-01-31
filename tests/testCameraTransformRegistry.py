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
import itertools
import unittest

import lsst.utils.tests
from lsst.pex.exceptions import LsstCppException
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as cameraGeom

class ConversionFunction(object):
    """Wrap a TransformRegistry conversion as a function(Point2D)->Point2D
    """
    def __init__(self, transformRegistry, fromSys, toSys):
        self.transformRegistry = transformRegistry
        self.fromSys = fromSys
        self.toSys = toSys

    def __call__(self, point):
        return self.transformRegistry.convert(point, self.fromSys, self.toSys)

class FuncPair(object):
    """Wrap a pair of function(Point2D)->Point2D functions as a single such function
    """
    def __init__(self, func1, func2):
        self.func1 = func1
        self.func2 = func2

    def __call__(self, point):
        return self.func2(self.func1(point))

def unityTransform(point):
    """Unity function(Point2D)->Point2D
    """
    return point


class CameraTransformRegistryTestCase(unittest.TestCase):
    def setUp(self):
        self.nativeSys = cameraGeom.FOCAL_PLANE
        self.pupilTransform = afwGeom.RadialXYTransform([0, 0.5, 0.01])
        transMap = {cameraGeom.PUPIL: self.pupilTransform}
        self.transReg = cameraGeom.CameraTransformRegistry(self.nativeSys, transMap)

    def tearDown(self):
        self.nativeSys = None
        self.pupilTransform = None
        self.transReg = None

    def compare2DFunctions(self, func1, func2, minVal=-10, maxVal=None, nVal=5):
        """Compare two functions(Point2D) -> Point2D over a range of values
        """
        if maxVal is None:
            maxVal = -minVal
        dVal = (maxVal - minVal) / (nVal - 1)
        for xInd in range(nVal):
            x = minVal + (xInd * dVal)
            for yInd in range(nVal):
                y = minVal + (yInd * dVal)
                fromPoint = afwGeom.Point2D(x, y)
                res1 = func1(fromPoint)
                res2 = func2(fromPoint)
                self.assertAlmostEqual(res1[0], res2[0])
                self.assertAlmostEqual(res1[1], res2[1])

    def testBasics(self):
        """Test basic attributes
        """
        for methodName in ("begin", "end", "constains", "size"):
            self.assertFalse(hasattr(self.transReg, methodName))

        self.assertTrue(self.nativeSys in self.transReg)
        self.assertTrue(cameraGeom.PUPIL in self.transReg)
        self.assertFalse(cameraGeom.CameraSys("garbage") in self.transReg)

        csList = self.transReg.getCoordSysList()
        self.assertTrue(len(csList) == 2)
        self.assertTrue(self.nativeSys in csList)
        self.assertTrue(cameraGeom.PUPIL in csList)


    def testIteration(self):
        """Test iteration, len and indexing
        """
        self.assertEquals(len(self.transReg), 2)

        csList = self.transReg.getCoordSysList()
        csList2 = [cs for cs in self.transReg]
        self.assertEquals(len(csList), len(self.transReg))
        self.assertEquals(tuple(csList), tuple(csList2))

        for cs in csList:
            xyTrans = self.transReg[cs]
            self.assertTrue(isinstance(xyTrans, afwGeom.XYTransform))

        self.assertRaises(LsstCppException, self.transReg.__getitem__, cameraGeom.CameraSys("missing"))

    def testTransforms(self):
        """Test that the contained transforms are the ones expected
        """
        nativeTr = self.transReg[self.nativeSys]
        self.compare2DFunctions(nativeTr.forwardTransform, unityTransform)
        self.compare2DFunctions(nativeTr.reverseTransform, unityTransform)

        pupilTr = self.transReg[cameraGeom.PUPIL]
        self.compare2DFunctions(pupilTr.forwardTransform, self.pupilTransform.forwardTransform)
        self.compare2DFunctions(pupilTr.reverseTransform, self.pupilTransform.reverseTransform)

    def testConvert(self):
        """Test convert
        """
        for fromSys in self.transReg.getCoordSysList():
            for toSys in self.transReg.getCoordSysList():
                trConvFunc = ConversionFunction(self.transReg, fromSys, toSys)
                if fromSys == toSys:
                    self.compare2DFunctions(trConvFunc, unityTransform)
                funcPair = FuncPair(
                    self.transReg[fromSys].forwardTransform,
                    self.transReg[toSys].reverseTransform,
                )
                self.compare2DFunctions(trConvFunc, funcPair)


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
