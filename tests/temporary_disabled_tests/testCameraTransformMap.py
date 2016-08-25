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
Tests for lsst.afw.cameraGeom.CameraTransformMap
"""
import itertools
import unittest

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as cameraGeom

class TransformWrapper(object):
    """Wrap a TransformMap transformation as a function(Point2D)->Point2D
    """
    def __init__(self, transformMap, fromSys, toSys):
        self.transformMap = transformMap
        self.fromSys = fromSys
        self.toSys = toSys

    def __call__(self, point):
        return self.transformMap.transform(point, self.fromSys, self.toSys)

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


class CameraTransformMapTestCase(unittest.TestCase):
    def setUp(self):
        self.nativeSys = cameraGeom.FOCAL_PLANE
        self.pupilTransform = afwGeom.RadialXYTransform([0, 0.5, 0.005])
        transforms = {cameraGeom.PUPIL: self.pupilTransform}
        self.transformMap = cameraGeom.CameraTransformMap(self.nativeSys, transforms)

    def tearDown(self):
        self.nativeSys = None
        self.pupilTransform = None
        self.transformMap = None

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
        for methodName in ("begin", "end", "contains", "size"):
            self.assertFalse(hasattr(self.transformMap, methodName))

        self.assertTrue(self.nativeSys in self.transformMap)
        self.assertTrue(cameraGeom.PUPIL in self.transformMap)
        self.assertFalse(cameraGeom.CameraSys("garbage") in self.transformMap)

        csList = self.transformMap.getCoordSysList()
        self.assertTrue(len(csList) == 2)
        self.assertTrue(self.nativeSys in csList)
        self.assertTrue(cameraGeom.PUPIL in csList)


    def testIteration(self):
        """Test iteration, len and indexing
        """
        self.assertEquals(len(self.transformMap), 2)

        csList = self.transformMap.getCoordSysList()
        csList2 = [cs for cs in self.transformMap]
        self.assertEquals(len(csList), len(self.transformMap))
        self.assertEquals(tuple(csList), tuple(csList2))

        for cs in csList:
            xyTrans = self.transformMap[cs]
            self.assertTrue(isinstance(xyTrans, afwGeom.XYTransform))

    def testGetItem(self):
        """Test that the contained transforms are the ones expected
        """
        nativeTr = self.transformMap[self.nativeSys]
        self.compare2DFunctions(nativeTr.forwardTransform, unityTransform)
        self.compare2DFunctions(nativeTr.reverseTransform, unityTransform)

        pupilTr = self.transformMap[cameraGeom.PUPIL]
        self.compare2DFunctions(pupilTr.forwardTransform, self.pupilTransform.forwardTransform)
        self.compare2DFunctions(pupilTr.reverseTransform, self.pupilTransform.reverseTransform)

        missingCamSys = cameraGeom.CameraSys("missing")
        self.assertRaises(lsst.pex.exceptions.Exception, self.transformMap.__getitem__, missingCamSys)

    def testGet(self):
        """Test the get method
        """
        for cs in self.transformMap.getCoordSysList():
            xyTrans2 = self.transformMap.get(cs)
            self.assertTrue(isinstance(xyTrans2, afwGeom.XYTransform))

        missingCamSys = cameraGeom.CameraSys("missing")
        shouldBeNone = self.transformMap.get(missingCamSys)
        self.assertTrue(shouldBeNone is None)
        self.assertRaises(Exception, self.transformMap.get, "badDataType")

        for default in (1, "hello", cameraGeom.CameraSys("default")):
            res = self.transformMap.get(missingCamSys, default)
            self.assertEquals(res, default)

    def testTransform(self):
        """Test transform method, point version
        """
        for fromSys in self.transformMap.getCoordSysList():
            for toSys in self.transformMap.getCoordSysList():
                trConvFunc = TransformWrapper(self.transformMap, fromSys, toSys)
                if fromSys == toSys:
                    self.compare2DFunctions(trConvFunc, unityTransform)
                funcPair = FuncPair(
                    self.transformMap[fromSys].reverseTransform,
                    self.transformMap[toSys].forwardTransform,
                )
                self.compare2DFunctions(trConvFunc, funcPair)

    def testTransformList(self):
        """Test transform method, list version
        """
        fromList = []
        for x in (-1.2, 0.0, 25.3):
            for y in (-23.4, 0.0, 2.3):
                fromList.append(afwGeom.Point2D(x, y))

        for fromSys in self.transformMap.getCoordSysList():
            for toSys in self.transformMap.getCoordSysList():
                toList = self.transformMap.transform(fromList, fromSys, toSys)

                self.assertEquals(len(fromList), len(toList))
                for fromPoint, toPoint in itertools.izip(fromList, toList):
                    predToPoint = self.transformMap.transform(fromPoint, fromSys, toSys)
                    for i in range(2):
                        self.assertAlmostEqual(predToPoint[i], toPoint[i])

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(CameraTransformMapTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
