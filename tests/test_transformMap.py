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
Tests for lsst.afw.cameraGeom.TransformMap
"""
import unittest

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.geom
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as cameraGeom


class TransformWrapper:
    """Wrap a TransformMap transformation as a function(Point2D)->Point2D
    """

    def __init__(self, transformMap, fromSys, toSys):
        self.transformMap = transformMap
        self.fromSys = fromSys
        self.toSys = toSys

    def __call__(self, point):
        return self.transformMap.transform(point, self.fromSys, self.toSys)


class Composition:
    """Wrap a pair of function(Point2D)->Point2D functions as a single
    function that calls the first function, then the second function on the
    result
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


class CameraTransformMapTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        self.nativeSys = cameraGeom.FOCAL_PLANE
        self.fieldTransform = afwGeom.makeRadialTransform([0, 0.5, 0.005])
        transforms = {cameraGeom.FIELD_ANGLE: self.fieldTransform}
        self.transformMap = cameraGeom.TransformMap(
            self.nativeSys, transforms)

    def tearDown(self):
        self.nativeSys = None
        self.fieldTransform = None
        self.transformMap = None

    def compare2DFunctions(self, func1, func2, minVal=-10, maxVal=None,
                           nVal=5):
        """Compare two functions(Point2D) -> Point2D over a range of values
        """
        if maxVal is None:
            maxVal = -minVal
        dVal = (maxVal - minVal) / (nVal - 1)
        for xInd in range(nVal):
            x = minVal + (xInd * dVal)
            for yInd in range(nVal):
                y = minVal + (yInd * dVal)
                fromPoint = lsst.geom.Point2D(x, y)
                res1 = func1(fromPoint)
                res2 = func2(fromPoint)
                self.assertPairsAlmostEqual(res1, res2)

    def testBasics(self):
        """Test basic attributes
        """
        for methodName in ("begin", "end", "contains", "size"):
            self.assertFalse(hasattr(self.transformMap, methodName))

        self.assertIn(self.nativeSys, self.transformMap)
        self.assertIn(cameraGeom.FIELD_ANGLE, self.transformMap)
        self.assertNotIn(cameraGeom.CameraSys("garbage"), self.transformMap)

        self.assertIn(self.nativeSys, self.transformMap)
        self.assertIn(cameraGeom.FIELD_ANGLE, self.transformMap)

    def testIteration(self):
        """Test iteration, len and indexing
        """
        self.assertEqual(len(self.transformMap), 2)

        systems = list(self.transformMap)
        self.assertEqual(len(systems), 2)

        for cs in systems:
            self.assertIsInstance(cs, cameraGeom.CameraSys)

    def testGetItem(self):
        """Test that the contained transforms are the ones expected
        """
        nativeTr = self.transformMap.getTransform(self.nativeSys,
                                                  self.nativeSys)
        self.compare2DFunctions(nativeTr.applyForward, unityTransform)
        self.compare2DFunctions(nativeTr.applyInverse, unityTransform)

        fieldTr = self.transformMap.getTransform(self.nativeSys,
                                                 cameraGeom.FIELD_ANGLE)
        self.compare2DFunctions(fieldTr.applyForward,
                                self.fieldTransform.applyForward)
        self.compare2DFunctions(fieldTr.applyInverse,
                                self.fieldTransform.applyInverse)

        fieldTrInv = self.transformMap.getTransform(cameraGeom.FIELD_ANGLE,
                                                    self.nativeSys)
        self.compare2DFunctions(fieldTrInv.applyForward,
                                self.fieldTransform.applyInverse)
        self.compare2DFunctions(fieldTrInv.applyInverse,
                                self.fieldTransform.applyForward)

        missingCamSys = cameraGeom.CameraSys("missing")
        with self.assertRaises(lsst.pex.exceptions.Exception):
            self.transformMap.getTransform(missingCamSys, self.nativeSys)
        with self.assertRaises(lsst.pex.exceptions.Exception):
            self.transformMap.getTransform(self.nativeSys, missingCamSys)

    def testTransform(self):
        """Test transform method, point version
        """
        for fromSys in self.transformMap:
            for toSys in self.transformMap:
                trConvFunc = TransformWrapper(
                    self.transformMap, fromSys, toSys)
                if fromSys == toSys:
                    self.compare2DFunctions(trConvFunc, unityTransform)
                funcPair = Composition(
                    self.transformMap
                        .getTransform(self.nativeSys, fromSys).applyInverse,
                    self.transformMap
                        .getTransform(self.nativeSys, toSys).applyForward
                )
                self.compare2DFunctions(trConvFunc, funcPair)

    def testTransformList(self):
        """Test transform method, list version
        """
        fromList = []
        for x in (-1.2, 0.0, 25.3):
            for y in (-23.4, 0.0, 2.3):
                fromList.append(lsst.geom.Point2D(x, y))

        for fromSys in self.transformMap:
            for toSys in self.transformMap:
                toList = self.transformMap.transform(fromList, fromSys, toSys)

                self.assertEqual(len(fromList), len(toList))
                for fromPoint, toPoint in zip(fromList, toList):
                    predToPoint = self.transformMap.transform(
                        fromPoint, fromSys, toSys)
                    self.assertPairsAlmostEqual(predToPoint, toPoint)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
