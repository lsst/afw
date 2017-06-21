"""
LSST Data Management System
See COPYRIGHT file at the top of the source tree.

This product includes software developed by the
LSST Project (http://www.lsst.org/).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the LSST License Statement and
the GNU General Public License along with this program. If not,
see <http://www.lsstcorp.org/LegalNotices/>.
"""
from __future__ import absolute_import, division, print_function
import unittest

from numpy.testing import assert_allclose, assert_equal
import astshim

import lsst.utils.tests
from lsst.afw.coord import IcrsCoord
import lsst.afw.geom as afwGeom
from lsst.pex.exceptions import InvalidParameterError
from lsst.afw.geom.testUtils import TransformTestBaseClass


class EndpointTestCase(TransformTestBaseClass):

    def setUp(self):
        self.longMessage = True

    def testIcrsCoordEndpoint(self):
        endpoint = afwGeom.IcrsCoordEndpoint()
        self.checkEndpointBasics(
            endpoint=endpoint, pointType=IcrsCoord, nAxes=2)
        self.assertEqual(repr(endpoint), "lsst.afw.geom.IcrsCoordEndpoint()")
        self.assertEqual("{}".format(endpoint), "IcrsCoordEndpoint()")

        for doPermute in (False, True):
            frame = astshim.SkyFrame()
            if doPermute:
                frame.permAxes([2, 1])
                self.assertEqual(frame.lonAxis, 2)
                self.assertEqual(frame.latAxis, 1)
            else:
                self.assertEqual(frame.lonAxis, 1)
                self.assertEqual(frame.latAxis, 2)
            endpoint.normalizeFrame(frame)
            # the normalized frame always has axis in order Lon, Lat
            self.assertEqual(frame.lonAxis, 1)
            self.assertEqual(frame.latAxis, 2)

        badFrame = astshim.Frame(2)
        with self.assertRaises(InvalidParameterError):
            endpoint.normalizeFrame(badFrame)

        newFrame = endpoint.makeFrame()
        self.assertEqual(type(newFrame), astshim.SkyFrame)
        self.assertEqual(newFrame.lonAxis, 1)
        self.assertEqual(newFrame.latAxis, 2)

    def testPoint2Endpoint(self):
        endpoint = afwGeom.Point2Endpoint()
        self.checkEndpointBasics(
            endpoint=endpoint, pointType=afwGeom.Point2D, nAxes=2)
        self.assertEqual(repr(endpoint), "lsst.afw.geom.Point2Endpoint()")
        self.assertEqual("{}".format(endpoint), "Point2Endpoint()")

        # normalize does not check the # of axes
        for n in range(4):
            frame1 = astshim.Frame(n)
            try:
                endpoint.normalizeFrame(frame1)
            except Exception as e:
                self.fail(
                    "endpoint.normalizeFrame(Frame({})) failed with error = {}".format(n, e))
        badFrame = astshim.SkyFrame()
        with self.assertRaises(InvalidParameterError):
            endpoint.normalizeFrame(badFrame)

    def testGenericEndpoint(self):
        for nAxes in (1, 2, 3, 4, 5):
            endpoint = afwGeom.GenericEndpoint(nAxes)
            self.checkEndpointBasics(
                endpoint=endpoint, pointType=list, nAxes=nAxes)
            self.assertEqual(
                repr(endpoint), "lsst.afw.geom.GenericEndpoint({})".format(nAxes))
            self.assertEqual("{}".format(endpoint),
                             "GenericEndpoint({})".format(nAxes))

            newFrame = endpoint.makeFrame()
            self.assertEqual(type(newFrame), astshim.Frame)
            self.assertEqual(newFrame.nAxes, nAxes)

        for nAxes in (-1, 0):
            with self.assertRaises(InvalidParameterError):
                afwGeom.GenericEndpoint(nAxes)

    def checkEndpointBasics(self, endpoint, pointType, nAxes):
        isAngle = pointType == IcrsCoord  # point components are Angles

        baseMsg = "endpoint={}, pointType={}, nAxes={}".format(
            endpoint, pointType, nAxes)

        self.assertEqual(endpoint.nAxes, nAxes, msg=baseMsg)

        # generate enough points to be interesting, but no need to overdo it
        nPoints = 4

        rawData = self.makeRawArrayData(nPoints=nPoints, nAxes=nAxes)
        pointList = endpoint.arrayFromData(rawData)
        self.assertEqual(endpoint.getNPoints(pointList), nPoints, msg=baseMsg)
        if isinstance(endpoint, afwGeom.GenericEndpoint):
            self.assertEqual(len(pointList[0]), nPoints, msg=baseMsg)
            assert_equal(rawData, pointList)
        else:
            self.assertEqual(len(pointList), nPoints, msg=baseMsg)
            for i, point in enumerate(pointList):
                for axis in range(nAxes):
                    msg = "{}, endpoint={}, i={}, point={}".format(
                        baseMsg, endpoint, i, point)
                    if isAngle:
                        desAngle = rawData[axis, i] * afwGeom.radians
                        self.assertAnglesAlmostEqual(
                            point[axis], desAngle, msg=msg)
                    else:
                        self.assertAlmostEqual(
                            point[axis], rawData[axis, i], msg=msg)

        rawDataRoundTrip = endpoint.dataFromArray(pointList)
        self.assertEqual(rawData.shape, rawDataRoundTrip.shape, msg=baseMsg)
        self.assertFloatsAlmostEqual(rawData, rawDataRoundTrip, msg=baseMsg)

        pointData = self.makeRawPointData(nAxes=nAxes)
        point = endpoint.pointFromData(pointData)
        self.assertEqual(type(point), pointType, msg=baseMsg)
        for axis in range(nAxes):
            msg = "{}, axis={}".format(baseMsg, axis)
            if isAngle:
                desAngle = pointData[axis] * afwGeom.radians
                self.assertAnglesAlmostEqual(point[axis], desAngle, msg=msg)
            else:
                self.assertAlmostEqual(point[axis], pointData[axis], msg=msg)

        pointDataRoundTrip = endpoint.dataFromPoint(point)
        assert_allclose(pointData, pointDataRoundTrip, err_msg=baseMsg)

    def testEndpointEquals(self):
        """Test Endpoint == Endpoint
        """
        for i1, point1 in enumerate(self.makeEndpoints()):
            for i2, point2 in enumerate(self.makeEndpoints()):
                if i1 == i2:
                    self.assertTrue(point1 == point2)
                    self.assertFalse(point1 != point2)
                else:
                    self.assertFalse(point1 == point2)
                    self.assertTrue(point1 != point2)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
