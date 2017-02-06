from __future__ import absolute_import, division
from builtins import range
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
import math

import lsst.utils.tests
from . import _Angle


__all__ = ["assertAnglesNearlyEqual", "assertPairsNearlyEqual", "assertBoxesNearlyEqual"]

@lsst.utils.tests.inTestCase
def assertAnglesNearlyEqual(testCase, ang0, ang1, maxDiff=0.001*_Angle.arcseconds,
        ignoreWrap=True, msg="Angles differ"):
    """!Assert that two angles are nearly equal, ignoring wrap differences by default

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
        measDiff.wrapCtr()
    if abs(measDiff) > maxDiff:
        testCase.fail("%s: measured difference %s arcsec > max allowed %s arcsec" %
            (msg, measDiff.asArcseconds(), maxDiff.asArcseconds()))

@lsst.utils.tests.inTestCase
def assertPairsNearlyEqual(testCase, pair0, pair1, maxDiff=1e-7, msg="Pairs differ"):
    """!Assert that two planar pairs (e.g. Point2D or Extent2D) are nearly equal

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
        testCase.fail("%s: measured radial distance = %s > maxDiff = %s" % (msg, measDiff, maxDiff))

@lsst.utils.tests.inTestCase
def assertBoxesNearlyEqual(testCase, box0, box1, maxDiff=1e-7, msg="Boxes differ"):
    """!Assert that two boxes (Box2D or Box2I) are nearly equal

    @warning Does not compare types, just compares values.

    @param[in] testCase  unittest.TestCase instance the test is part of;
                        an object supporting one method: fail(self, msgStr)
    @param[in] box0  box 0
    @param[in] box1  box 1
    @param[in] maxDiff  maximum radial separation between the min points and max points
    @param[in] msg  exception message prefix; details of the error are appended after ": "

    @throw AssertionError if the radial difference of the min points or max points is greater than maxDiff
    """
    assertPairsNearlyEqual(testCase, box0.getMin(), box1.getMin(), maxDiff=maxDiff, msg=msg + ": min")
    assertPairsNearlyEqual(testCase, box0.getMax(), box1.getMax(), maxDiff=maxDiff, msg=msg + ": max")
