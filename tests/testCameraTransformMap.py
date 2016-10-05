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
#pybind11#Tests for lsst.afw.cameraGeom.CameraTransformMap
#pybind11#"""
#pybind11#import itertools
#pybind11#import unittest
#pybind11#from builtins import zip
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.cameraGeom as cameraGeom
#pybind11#
#pybind11#
#pybind11#class TransformWrapper(object):
#pybind11#    """Wrap a TransformMap transformation as a function(Point2D)->Point2D
#pybind11#    """
#pybind11#
#pybind11#    def __init__(self, transformMap, fromSys, toSys):
#pybind11#        self.transformMap = transformMap
#pybind11#        self.fromSys = fromSys
#pybind11#        self.toSys = toSys
#pybind11#
#pybind11#    def __call__(self, point):
#pybind11#        return self.transformMap.transform(point, self.fromSys, self.toSys)
#pybind11#
#pybind11#
#pybind11#class FuncPair(object):
#pybind11#    """Wrap a pair of function(Point2D)->Point2D functions as a single such function
#pybind11#    """
#pybind11#
#pybind11#    def __init__(self, func1, func2):
#pybind11#        self.func1 = func1
#pybind11#        self.func2 = func2
#pybind11#
#pybind11#    def __call__(self, point):
#pybind11#        return self.func2(self.func1(point))
#pybind11#
#pybind11#
#pybind11#def unityTransform(point):
#pybind11#    """Unity function(Point2D)->Point2D
#pybind11#    """
#pybind11#    return point
#pybind11#
#pybind11#
#pybind11#class CameraTransformMapTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.nativeSys = cameraGeom.FOCAL_PLANE
#pybind11#        self.pupilTransform = afwGeom.RadialXYTransform([0, 0.5, 0.005])
#pybind11#        transforms = {cameraGeom.PUPIL: self.pupilTransform}
#pybind11#        self.transformMap = cameraGeom.CameraTransformMap(self.nativeSys, transforms)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        self.nativeSys = None
#pybind11#        self.pupilTransform = None
#pybind11#        self.transformMap = None
#pybind11#
#pybind11#    def compare2DFunctions(self, func1, func2, minVal=-10, maxVal=None, nVal=5):
#pybind11#        """Compare two functions(Point2D) -> Point2D over a range of values
#pybind11#        """
#pybind11#        if maxVal is None:
#pybind11#            maxVal = -minVal
#pybind11#        dVal = (maxVal - minVal) / (nVal - 1)
#pybind11#        for xInd in range(nVal):
#pybind11#            x = minVal + (xInd * dVal)
#pybind11#            for yInd in range(nVal):
#pybind11#                y = minVal + (yInd * dVal)
#pybind11#                fromPoint = afwGeom.Point2D(x, y)
#pybind11#                res1 = func1(fromPoint)
#pybind11#                res2 = func2(fromPoint)
#pybind11#                self.assertAlmostEqual(res1[0], res2[0])
#pybind11#                self.assertAlmostEqual(res1[1], res2[1])
#pybind11#
#pybind11#    def testBasics(self):
#pybind11#        """Test basic attributes
#pybind11#        """
#pybind11#        for methodName in ("begin", "end", "contains", "size"):
#pybind11#            self.assertFalse(hasattr(self.transformMap, methodName))
#pybind11#
#pybind11#        self.assertIn(self.nativeSys, self.transformMap)
#pybind11#        self.assertIn(cameraGeom.PUPIL, self.transformMap)
#pybind11#        self.assertNotIn(cameraGeom.CameraSys("garbage"), self.transformMap)
#pybind11#
#pybind11#        csList = self.transformMap.getCoordSysList()
#pybind11#        self.assertEqual(len(csList), 2)
#pybind11#        self.assertIn(self.nativeSys, csList)
#pybind11#        self.assertIn(cameraGeom.PUPIL, csList)
#pybind11#
#pybind11#    def testIteration(self):
#pybind11#        """Test iteration, len and indexing
#pybind11#        """
#pybind11#        self.assertEqual(len(self.transformMap), 2)
#pybind11#
#pybind11#        csList = self.transformMap.getCoordSysList()
#pybind11#        csList2 = [cs for cs in self.transformMap]
#pybind11#        self.assertEqual(len(csList), len(self.transformMap))
#pybind11#        self.assertEqual(tuple(csList), tuple(csList2))
#pybind11#
#pybind11#        for cs in csList:
#pybind11#            xyTrans = self.transformMap[cs]
#pybind11#            self.assertIsInstance(xyTrans, afwGeom.XYTransform)
#pybind11#
#pybind11#    def testGetItem(self):
#pybind11#        """Test that the contained transforms are the ones expected
#pybind11#        """
#pybind11#        nativeTr = self.transformMap[self.nativeSys]
#pybind11#        self.compare2DFunctions(nativeTr.forwardTransform, unityTransform)
#pybind11#        self.compare2DFunctions(nativeTr.reverseTransform, unityTransform)
#pybind11#
#pybind11#        pupilTr = self.transformMap[cameraGeom.PUPIL]
#pybind11#        self.compare2DFunctions(pupilTr.forwardTransform, self.pupilTransform.forwardTransform)
#pybind11#        self.compare2DFunctions(pupilTr.reverseTransform, self.pupilTransform.reverseTransform)
#pybind11#
#pybind11#        missingCamSys = cameraGeom.CameraSys("missing")
#pybind11#        with self.assertRaises(lsst.pex.exceptions.Exception):
#pybind11#            self.transformMap.__getitem__(missingCamSys)
#pybind11#
#pybind11#    def testGet(self):
#pybind11#        """Test the get method
#pybind11#        """
#pybind11#        for cs in self.transformMap.getCoordSysList():
#pybind11#            xyTrans2 = self.transformMap.get(cs)
#pybind11#            self.assertIsInstance(xyTrans2, afwGeom.XYTransform)
#pybind11#
#pybind11#        missingCamSys = cameraGeom.CameraSys("missing")
#pybind11#        shouldBeNone = self.transformMap.get(missingCamSys)
#pybind11#        self.assertIsNone(shouldBeNone)
#pybind11#        with self.assertRaises(Exception):
#pybind11#            self.transformMap.get("badDataType")
#pybind11#
#pybind11#        for default in (1, "hello", cameraGeom.CameraSys("default")):
#pybind11#            res = self.transformMap.get(missingCamSys, default)
#pybind11#            self.assertEqual(res, default)
#pybind11#
#pybind11#    def testTransform(self):
#pybind11#        """Test transform method, point version
#pybind11#        """
#pybind11#        for fromSys in self.transformMap.getCoordSysList():
#pybind11#            for toSys in self.transformMap.getCoordSysList():
#pybind11#                trConvFunc = TransformWrapper(self.transformMap, fromSys, toSys)
#pybind11#                if fromSys == toSys:
#pybind11#                    self.compare2DFunctions(trConvFunc, unityTransform)
#pybind11#                funcPair = FuncPair(
#pybind11#                    self.transformMap[fromSys].reverseTransform,
#pybind11#                    self.transformMap[toSys].forwardTransform,
#pybind11#                )
#pybind11#                self.compare2DFunctions(trConvFunc, funcPair)
#pybind11#
#pybind11#    def testTransformList(self):
#pybind11#        """Test transform method, list version
#pybind11#        """
#pybind11#        fromList = []
#pybind11#        for x in (-1.2, 0.0, 25.3):
#pybind11#            for y in (-23.4, 0.0, 2.3):
#pybind11#                fromList.append(afwGeom.Point2D(x, y))
#pybind11#
#pybind11#        for fromSys in self.transformMap.getCoordSysList():
#pybind11#            for toSys in self.transformMap.getCoordSysList():
#pybind11#                toList = self.transformMap.transform(fromList, fromSys, toSys)
#pybind11#
#pybind11#                self.assertEqual(len(fromList), len(toList))
#pybind11#                for fromPoint, toPoint in zip(fromList, toList):
#pybind11#                    predToPoint = self.transformMap.transform(fromPoint, fromSys, toSys)
#pybind11#                    for i in range(2):
#pybind11#                        self.assertAlmostEqual(predToPoint[i], toPoint[i])
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
