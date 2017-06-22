from __future__ import absolute_import, division, print_function
#
# LSST Data Management System
# Copyright 2015 LSST Corporation.
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
"""Utilities that should be imported into the lsst.afw.geom namespace when lsst.afw.geom is used

In the case of the assert functions, importing them makes them available in lsst.utils.tests.TestCase
"""
__all__ = ["assertAnglesAlmostEqual", "assertPairsAlmostEqual",
           "assertPairListsAlmostEqual", "assertSpherePointsAlmostEqual",
           "assertSpherePointListsAlmostEqual", "assertBoxesAlmostEqual",
           "makeEndpoints", "assertAnglesNearlyEqual", "assertPairsNearlyEqual",
           "assertBoxesNearlyEqual"]

from builtins import range
import warnings
import math

import numpy as np

import lsst.utils.tests
from .angle import arcseconds
from .endpoint import GenericEndpoint, Point2Endpoint, IcrsCoordEndpoint


def extraMsg(msg):
    """Format extra error message, if any
    """
    if msg:
        return ": " + msg
    return ""


@lsst.utils.tests.inTestCase
def assertAnglesAlmostEqual(testCase, ang0, ang1, maxDiff=0.001*arcseconds,
                            ignoreWrap=True, msg="Angles differ"):
    """!Assert that two angles are almost equal, ignoring wrap differences by default

    @param[in] testCase  unittest.TestCase instance the test is part of;
                        an object supporting one method: fail(self, msgStr)
    @param[in] ang0  angle 0 (an lsst.afw.geom.Angle)
    @param[in] ang1  angle 1 (an lsst.afw.geom.Angle)
    @param[in] maxDiff  maximum difference between the two angles (an lsst.afw.geom.Angle)
    @param[in] ignoreWrap  ignore wrap when comparing the angles?
        - if True then wrap is ignored, e.g. 0 and 360 degrees are considered equal
        - if False then wrap matters, e.g. 0 and 360 degrees are considered different
    @param[in] msg  exception message prefix; details of the error are appended after ": "

    @throw AssertionError if the difference is greater than maxDiff
    """
    measDiff = ang1 - ang0
    if ignoreWrap:
        measDiff = measDiff.wrapCtr()
    if abs(measDiff) > maxDiff:
        testCase.fail("%s: measured difference %s arcsec > max allowed %s arcsec" %
                      (msg, measDiff.asArcseconds(), maxDiff.asArcseconds()))


@lsst.utils.tests.inTestCase
def assertPairsAlmostEqual(testCase, pair0, pair1, maxDiff=1e-7, msg="Pairs differ"):
    """!Assert that two Cartesian points are almost equal.

    Each point can be any indexable pair of two floats, including
    Point2D or Extent2D, a list or a tuple.

    @warning Does not compare types, just compares values.

    @param[in] testCase  unittest.TestCase instance the test is part of;
                        an object supporting one method: fail(self, msgStr)
    @param[in] pair0  pair 0 (a pair of floats)
    @param[in] pair1  pair 1 (a pair of floats)
    @param[in] maxDiff  maximum radial separation between the two points
    @param[in] msg  exception message prefix; details of the error are appended after ": "

    @throw AssertionError if the radial difference is greater than maxDiff
    """
    if len(pair0) != 2:
        raise RuntimeError("len(pair0)=%s != 2" % (len(pair0),))
    if len(pair1) != 2:
        raise RuntimeError("len(pair1)=%s != 2" % (len(pair1),))

    pairDiff = [float(pair1[i] - pair0[i]) for i in range(2)]
    measDiff = math.hypot(*pairDiff)
    if measDiff > maxDiff:
        testCase.fail("%s: measured radial distance = %s > maxDiff = %s" % (
            msg, measDiff, maxDiff))


@lsst.utils.tests.inTestCase
def assertPairListsAlmostEqual(testCase, list0, list1, maxDiff=1e-7, msg=None):
    """!Assert that two lists of Cartesian points are almost equal

    Each point can be any indexable pair of two floats, including
    Point2D or Extent2D, a list or a tuple.

    @warning Does not compare types, just values.

    @param[in] testCase  unittest.TestCase instance the test is part of;
                        an object supporting one method: fail(self, msgStr)
    @param[in] list0  list of pairs 0 (each element a pair of floats)
    @param[in] list1  list of pairs 1
    @param[in] maxDiff  maximum radial separation between the two points
    @param[in] msg  additional information for the error message; appended after ": "

    @throw AssertionError if the radial difference is greater than maxDiff
    """
    testCase.assertEqual(len(list0), len(list1))
    lenList1 = np.array([len(val) for val in list0])
    lenList2 = np.array([len(val) for val in list1])
    testCase.assertTrue(np.all(lenList1 == 2))
    testCase.assertTrue(np.all(lenList2 == 2))

    diffArr = np.array([(val0[0] - val1[0], val0[1] - val1[1])
                        for val0, val1 in zip(list0, list1)], dtype=float)
    sepArr = np.hypot(diffArr[:, 0], diffArr[:, 1])
    badArr = sepArr > maxDiff
    if np.any(badArr):
        maxInd = np.argmax(sepArr)
        testCase.fail("PairLists differ in %s places; max separation is at %s: %s > %s%s" %
                      (np.sum(badArr), maxInd, sepArr[maxInd], maxDiff, extraMsg(msg)))


@lsst.utils.tests.inTestCase
def assertSpherePointsAlmostEqual(testCase, sp0, sp1, maxSep=0.001*arcseconds, msg=""):
    """!Assert that two SpherePoints are almost equal

    @param[in] testCase  unittest.TestCase instance the test is part of;
                        an object supporting one method: fail(self, msgStr)
    @param[in] sp0  SpherePoint 0
    @param[in] sp1  SpherePoint 1
    @param[in] maxSep  maximum separation, an lsst.afw.geom.Angle
    @param[in] msg  extra information to be printed with any error message
    """
    if sp0.separation(sp1) > maxSep:
        testCase.fail("Angular separation between %s and %s = %s\" > maxSep = %s\"%s" %
                      (sp0, sp1, sp0.separation(sp1).asArcseconds(), maxSep.asArcseconds(), extraMsg(msg)))


@lsst.utils.tests.inTestCase
def assertSpherePointListsAlmostEqual(testCase, splist0, splist1, maxSep=0.001*arcseconds, msg=None):
    """!Assert that two lists of SpherePoints are almost equal

    @param[in] testCase  unittest.TestCase instance the test is part of;
                        an object supporting one method: fail(self, msgStr)
    @param[in] splist0  list of SpherePoints 0
    @param[in] splist1  list of SpherePoints 1
    @param[in] maxSep  maximum separation, an lsst.afw.geom.Angle
    @param[in] msg  exception message prefix; details of the error are appended after ": "
    """
    testCase.assertEqual(len(splist0), len(splist1), msg=msg)
    sepArr = np.array([sp0.separation(sp1)
                       for sp0, sp1 in zip(splist0, splist1)])
    badArr = sepArr > maxSep
    if np.any(badArr):
        maxInd = np.argmax(sepArr)
        testCase.fail("SpherePointLists differ in %s places; max separation is at %s: %s\" > %s\"%s" %
                      (np.sum(badArr), maxInd, sepArr[maxInd].asArcseconds(),
                       maxSep.asArcseconds(), extraMsg(msg)))


@lsst.utils.tests.inTestCase
def assertBoxesAlmostEqual(testCase, box0, box1, maxDiff=1e-7, msg="Boxes differ"):
    """!Assert that two boxes (Box2D or Box2I) are almost equal

    @warning Does not compare types, just compares values.

    @param[in] testCase  unittest.TestCase instance the test is part of;
                        an object supporting one method: fail(self, msgStr)
    @param[in] box0  box 0
    @param[in] box1  box 1
    @param[in] maxDiff  maximum radial separation between the min points and max points
    @param[in] msg  exception message prefix; details of the error are appended after ": "

    @throw AssertionError if the radial difference of the min points or max points is greater than maxDiff
    """
    assertPairsAlmostEqual(testCase, box0.getMin(),
                           box1.getMin(), maxDiff=maxDiff, msg=msg + ": min")
    assertPairsAlmostEqual(testCase, box0.getMax(),
                           box1.getMax(), maxDiff=maxDiff, msg=msg + ": max")


@lsst.utils.tests.inTestCase
def makeEndpoints(testCase):
    """Generate a representative sample of Endpoints.

    Returns
    -------
    x : `list`
        List of endpoints with enough diversity to exercise Endpoint-related
        code. Each invocation of this method shall return independent objects.
    """
    return [GenericEndpoint(n) for n in range(1, 6)] + \
           [Point2Endpoint(), IcrsCoordEndpoint()]


@lsst.utils.tests.inTestCase
def assertAnglesNearlyEqual(*args, **kwargs):
    warnings.warn("Deprecated. Use assertAnglesAlmostEqual",
                  DeprecationWarning)
    assertAnglesAlmostEqual(*args, **kwargs)


@lsst.utils.tests.inTestCase
def assertPairsNearlyEqual(*args, **kwargs):
    warnings.warn("Deprecated. Use assertPairsAlmostEqual", DeprecationWarning)
    assertPairsAlmostEqual(*args, **kwargs)


@lsst.utils.tests.inTestCase
def assertBoxesNearlyEqual(*args, **kwargs):
    warnings.warn("Deprecated. Use assertBoxesAlmostEqual", DeprecationWarning)
    assertBoxesAlmostEqual(*args, **kwargs)
