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

import numpy as np
import astshim

import lsst.utils.tests
import lsst.afw.geom as afwGeom
from lsst.pex.exceptions import InvalidParameterError


def makeRawArrayData(nPoints, nAxes, delta=0.123):
    return np.array([[i + j*delta for j in range(nAxes)] for i in range(nPoints)])


def makeRawPointData(nAxes, delta=0.123):
    return [i*delta for i in range(nAxes)]

# names of enpoints
NameList = ("Generic", "Point2", "Point3", "SpherePoint")


def makeCoeffs(nIn, nOut):
    """Make an array of coefficients for astshim.PolyMap for the following equation:

    fj(x) = C0j x0 + C1j x1 + C2j x2...
    where:
    * i ranges from 0 to nIn-1
    * j ranges from 0 to nOut-1,
    * Cij = 0.001 i (j+1)
    """
    baseCoeff = 0.001
    forwardCoeffs = []
    for out_ind in range(nOut):
        coeffOffset = baseCoeff * out_ind
        for in_ind in range(nIn):
            coeff = baseCoeff * (in_ind + 1) + coeffOffset
            coeffArr = [coeff, out_ind + 1] + [1 if i == in_ind else 0 for i in range(nIn)]
            forwardCoeffs.append(coeffArr)
    return np.array(forwardCoeffs, dtype=float)


def makePolyMap(nIn, nOut):
    """Make an astShim.PolyMap suitable for testing

    The forward transform is as follows:
    fj(x) = C0j x0 + C1j x1 + C2j x2... where Cij = 0.001 i (j+1)

    The reverse transform is the same equation with i and j reversed
    thus it is NOT the inverse of the forward direction,
    but is something that can be easily evaluated.

    The equation is chosen for the following reasons:
    - It is well defined for any value of nIn, nOut
    - It stays small for small x, to avoid wraparound of angles for SpherePoint endpoints
    """
    forwardCoeffs = makeCoeffs(nIn, nOut)
    reverseCoeffs = makeCoeffs(nOut, nIn)
    polyMap = astshim.PolyMap(forwardCoeffs, reverseCoeffs)
    assert polyMap.getNin() == nIn
    assert polyMap.getNout() == nOut
    return polyMap


class TransformTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        self.longMessage = True

        # GoodNaxes is dict of endpoint class name prefix:
        #    tuple containing 0 or more valid numbers of axes
        self.goodNaxes = {
            "Generic": (1, 2, 3, 4),  # all numbers of axes are valid for GenericEndpoint
            "Point2": (2,),
            "Point3": (3,),
            "SpherePoint": (2,),
        }

        # BadAxes is dict of endpoint class name prefix:
        #    tuple containing 0 or more invalid numbers of axes
        self.badNaxes = {
            "Generic": (),  # all numbers of axes are valid for GenericEndpoint
            "Point2": (1, 3, 4),
            "Point3": (1, 2, 4),
            "SpherePoint": (1, 3, 4),
        }

    def checkTransform(self, fromName, toName):
        """Check a Transform_<fromName>_<toName>

        fromName: one of Namelist
        toName  one of NameList
        fromAxes  number of axes in fromFrame
        toAxes  number of axes in toFrame
        """

        transformClassName = "Transform{}To{}".format(fromName, toName)
        transformClass = getattr(afwGeom, transformClassName)
        baseMsg = "transformClass={}".format(transformClass.__name__)
        for nIn in self.goodNaxes[fromName]:
            # check invalid numbers of output for the given toName
            for badNout in self.badNaxes[toName]:
                badPolyMap = makePolyMap(nIn, badNout)
                msg = "{}, nIn={}, badNout={}".format(baseMsg, nIn, badNout)
                with self.assertRaises(InvalidParameterError, msg=msg):
                    transformClass(badPolyMap)

            # check valid numbers of outputs for the given toName
            for nOut in self.goodNaxes[toName]:
                msg = "{}, nIn={}, nOut={}".format(baseMsg, nIn, nOut)
                polyMap = makePolyMap(nIn, nOut)
                transform = transformClass(polyMap)

                desStr = "{}[{}->{}]".format(transformClassName, nIn, nOut)
                self.assertEqual("{}".format(transform), desStr)
                self.assertEqual(repr(transform), "lsst.afw.geom." + desStr)

                fromEndpoint = transform.getFromEndpoint()
                toEndpoint = transform.getToEndpoint()
                frameSet = transform.getFrameSet()

                # forward transformation of one point
                rawInPoint = makeRawPointData(nIn)
                inPoint = fromEndpoint.pointFromData(rawInPoint)
                outPoint = transform.tranForward(inPoint)
                rawOutPoint = toEndpoint.dataFromPoint(outPoint)
                # TODO replace all use of np.allclose with assertFloatsAlmostEqual once DM-9707 is fixed
                self.assertTrue(np.allclose(rawOutPoint, polyMap.tranForward(rawInPoint)), msg=msg)
                self.assertTrue(np.allclose(rawOutPoint, frameSet.tranForward(rawInPoint)), msg=msg)

                # inverse transformation of one point;
                # remember that the inverse will not give the original values!
                inversePoint = transform.tranInverse(outPoint)
                rawInversePoint = fromEndpoint.dataFromPoint(inversePoint)
                self.assertTrue(np.allclose(rawInversePoint, polyMap.tranInverse(rawOutPoint)), msg=msg)
                self.assertTrue(np.allclose(rawInversePoint, frameSet.tranInverse(rawOutPoint)), msg=msg)

                # forward transformation of an array of points
                nPoints = 7  # arbitrary
                rawInArray = makeRawArrayData(nPoints, nIn)
                inArray = fromEndpoint.arrayFromData(rawInArray)
                outArray = transform.tranForward(inArray)
                rawOutArray = toEndpoint.dataFromArray(outArray)
                self.assertFloatsAlmostEqual(rawOutArray, polyMap.tranForward(rawInArray), msg=msg)
                self.assertFloatsAlmostEqual(rawOutArray, frameSet.tranForward(rawInArray), msg=msg)

                # inverse transformation of an array of points;
                # remember that the inverse will not give the original values!
                inverseArray = transform.tranInverse(outArray)
                rawInverseArray = fromEndpoint.dataFromArray(inverseArray)
                self.assertFloatsAlmostEqual(rawInverseArray, polyMap.tranInverse(rawOutArray), msg=msg)
                self.assertFloatsAlmostEqual(rawInverseArray, frameSet.tranInverse(rawOutArray), msg=msg)

        # check invalid numbers of inputs with valid and invalid #s of inputs
        for badNin in self.badNaxes[fromName]:
            for nOut in self.goodNaxes[toName] + self.badNaxes[toName]:
                badPolyMap = makePolyMap(badNin, nOut)
                msg = "{}, badNin={}, nOut={}".format(baseMsg, nIn, nOut)
                with self.assertRaises(InvalidParameterError, msg=msg):
                    transformClass(badPolyMap)

    def testTransforms(self):
        for fromName in NameList:
            for toName in NameList:
                self.checkTransform(fromName, toName)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
