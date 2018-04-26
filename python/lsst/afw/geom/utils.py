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
__all__ = ["wcsAlmostEqualOverBBox"]

import itertools
import warnings
import math

import numpy as np

import lsst.utils.tests
from .angle import arcseconds
from .box import Box2D
from .coordinates import Point2D
from .endpoint import GenericEndpoint, Point2Endpoint, SpherePointEndpoint


def extraMsg(msg):
    """Format extra error message, if any
    """
    if msg:
        return ": " + msg
    return ""


@lsst.utils.tests.inTestCase
def assertAnglesAlmostEqual(testCase, ang0, ang1, maxDiff=0.001*arcseconds,
                            ignoreWrap=True, msg="Angles differ"):
    r"""Assert that two `~lsst.afw.geom.Angle`\ s are almost equal, ignoring wrap differences by default

    Parameters
    ----------
    testCase : `unittest.TestCase`
        test case the test is part of; an object supporting one method: fail(self, msgStr)
    ang0 : `lsst.afw.geom.Angle`
        angle 0
    ang1 : `an lsst.afw.geom.Angle`
        angle 1
    maxDiff : `an lsst.afw.geom.Angle`
        maximum difference between the two angles
    ignoreWrap : `bool`
        ignore wrap when comparing the angles?
        - if True then wrap is ignored, e.g. 0 and 360 degrees are considered equal
        - if False then wrap matters, e.g. 0 and 360 degrees are considered different
    msg : `str`
        exception message prefix; details of the error are appended after ": "

    Raises
    ------
    AssertionError
        Raised if the difference is greater than ``maxDiff``
    """
    measDiff = ang1 - ang0
    if ignoreWrap:
        measDiff = measDiff.wrapCtr()
    if abs(measDiff) > maxDiff:
        testCase.fail("%s: measured difference %s arcsec > max allowed %s arcsec" %
                      (msg, measDiff.asArcseconds(), maxDiff.asArcseconds()))


@lsst.utils.tests.inTestCase
def assertPairsAlmostEqual(testCase, pair0, pair1, maxDiff=1e-7, msg="Pairs differ"):
    """Assert that two Cartesian points are almost equal.

    Each point can be any indexable pair of two floats, including
    Point2D or Extent2D, a list or a tuple.

    Parameters
    ----------
    testCase : `unittest.TestCase`
        test case the test is part of; an object supporting one method: fail(self, msgStr)
    pair0 : pair of `float`
        pair 0
    pair1 : pair of `floats`
        pair 1
    maxDiff : `float`
        maximum radial separation between the two points
    msg : `str`
        exception message prefix; details of the error are appended after ": "

    Raises
    ------
    AssertionError
        Raised if the radial difference is greater than ``maxDiff``

    Notes
    -----
    .. warning::

       Does not compare types, just compares values.
    """
    if len(pair0) != 2:
        raise RuntimeError("len(pair0)=%s != 2" % (len(pair0),))
    if len(pair1) != 2:
        raise RuntimeError("len(pair1)=%s != 2" % (len(pair1),))

    pairDiff = [float(pair1[i] - pair0[i]) for i in range(2)]
    measDiff = math.hypot(*pairDiff)
    if measDiff > maxDiff:
        testCase.fail("%s: measured radial distance = %s > maxDiff = %s, pair0=(%r, %r), pair1=(%r, %r)" %
                      (msg, measDiff, maxDiff, pair0[0], pair0[1], pair1[0], pair1[1]))


@lsst.utils.tests.inTestCase
def assertPairListsAlmostEqual(testCase, list0, list1, maxDiff=1e-7, msg=None):
    """Assert that two lists of Cartesian points are almost equal

    Each point can be any indexable pair of two floats, including
    Point2D or Extent2D, a list or a tuple.

    Parameters
    ----------
    testCase : `unittest.TestCase`
        test case the test is part of; an object supporting one method: fail(self, msgStr)
    list0 : `list` of pairs of `float`
        list of pairs 0
    list1 : `list` of pairs of `float`
        list of pairs 1
    maxDiff : `float`
        maximum radial separation between the two points
    msg : `str`
        additional information for the error message; appended after ": "

    Raises
    ------
    AssertionError
        Raised if the radial difference is greater than ``maxDiff``

    Notes
    -----
    .. warning::

       Does not compare types, just values.
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
    r"""Assert that two `~lsst.afw.geom.SpherePoint`\ s are almost equal

    Parameters
    ----------
    testCase : `unittest.TestCase`
        test case the test is part of; an object supporting one method: fail(self, msgStr)
    sp0 : `lsst.afw.geom.SpherePoint`
        SpherePoint 0
    sp1 : `lsst.afw.geom.SpherePoint`
        SpherePoint 1
    maxSep : `lsst.afw.geom.Angle`
        maximum separation
    msg : `str`
        extra information to be printed with any error message
    """
    if sp0.separation(sp1) > maxSep:
        testCase.fail("Angular separation between %s and %s = %s\" > maxSep = %s\"%s" %
                      (sp0, sp1, sp0.separation(sp1).asArcseconds(), maxSep.asArcseconds(), extraMsg(msg)))


@lsst.utils.tests.inTestCase
def assertSpherePointListsAlmostEqual(testCase, splist0, splist1, maxSep=0.001*arcseconds, msg=None):
    r"""Assert that two lists of `~lsst.afw.geom.SpherePoint`\ s are almost equal

    Parameters
    ----------
    testCase : `unittest.TestCase`
        test case the test is part of; an object supporting one method: fail(self, msgStr)
    splist0 : `list` of `lsst.afw.geom.SpherePoint`
        list of SpherePoints 0
    splist1 : `list` of `lsst.afw.geom.SpherePoint`
        list of SpherePoints 1
    maxSep : `lsst.afw.geom.Angle`
        maximum separation
    msg : `str`
        exception message prefix; details of the error are appended after ": "
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
    """Assert that two boxes (`~lsst.afw.geom.Box2D` or `~lsst.afw.geom.Box2I`) are almost equal

    Parameters
    ----------
    testCase : `unittest.TestCase`
        test case the test is part of; an object supporting one method: fail(self, msgStr)
    box0 : `lsst.afw.geom.Box2D` or `lsst.afw.geom.Box2I`
        box 0
    box1 : `lsst.afw.geom.Box2D` or `lsst.afw.geom.Box2I`
        box 1
    maxDiff : `float`
        maximum radial separation between the min points and max points
    msg : `str`
        exception message prefix; details of the error are appended after ": "

    Raises
    ------
    AssertionError
        Raised if the radial difference of the min points or max points is greater than maxDiff

    Notes
    -----
    .. warning::

       Does not compare types, just compares values.
    """
    assertPairsAlmostEqual(testCase, box0.getMin(),
                           box1.getMin(), maxDiff=maxDiff, msg=msg + ": min")
    assertPairsAlmostEqual(testCase, box0.getMax(),
                           box1.getMax(), maxDiff=maxDiff, msg=msg + ": max")


def _compareWcsOverBBox(wcs0, wcs1, bbox, maxDiffSky=0.01*arcseconds,
                        maxDiffPix=0.01, nx=5, ny=5, doShortCircuit=True):
    """Compare two :py:class:`WCS <lsst.afw.geom.SkyWcs>` over a rectangular grid of pixel positions

    Parameters
    ----------
    wcs0 : `lsst.afw.geom.SkyWcs`
        WCS 0
    wcs1 : `lsst.afw.geom.SkyWcs`
        WCS 1
    bbox : `lsst.afw.geom.Box2I` or `lsst.afw.geom.Box2D`
        boundaries of pixel grid over which to compare the WCSs
    maxDiffSky : `lsst.afw.geom.Angle`
        maximum separation between sky positions computed using Wcs.pixelToSky
    maxDiffPix : `float`
        maximum separation between pixel positions computed using Wcs.skyToPixel
    nx : `int`
        number of points in x for the grid of pixel positions
    ny : `int`
        number of points in y for the grid of pixel positions
    doShortCircuit : `bool`
        if True then stop at the first error, else test all values in the grid
        and return information about the worst violations found

    Returns
    -------
    msg : `str`
        an empty string if the WCS are sufficiently close; else return a string describing
        the largest error measured in pixel coordinates (if sky to pixel error was excessive)
        and sky coordinates (if pixel to sky error was excessive). If doShortCircuit is true
        then the reported error is likely to be much less than the maximum error across the
        whole pixel grid.
    """
    if nx < 1 or ny < 1:
        raise RuntimeError(
            "nx = %s and ny = %s must both be positive" % (nx, ny))
    if maxDiffSky <= 0*arcseconds:
        raise RuntimeError("maxDiffSky = %s must be positive" % (maxDiffSky,))
    if maxDiffPix <= 0:
        raise RuntimeError("maxDiffPix = %s must be positive" % (maxDiffPix,))

    bboxd = Box2D(bbox)
    xList = np.linspace(bboxd.getMinX(), bboxd.getMaxX(), nx)
    yList = np.linspace(bboxd.getMinY(), bboxd.getMaxY(), ny)
    # we don't care about measured error unless it is too large, so initialize
    # to max allowed
    measDiffSky = (maxDiffSky, "?")  # (sky diff, pix pos)
    measDiffPix = (maxDiffPix, "?")  # (pix diff, sky pos)
    for x, y in itertools.product(xList, yList):
        fromPixPos = Point2D(x, y)
        sky0 = wcs0.pixelToSky(fromPixPos)
        sky1 = wcs1.pixelToSky(fromPixPos)
        diffSky = sky0.separation(sky1)
        if diffSky > measDiffSky[0]:
            measDiffSky = (diffSky, fromPixPos)
            if doShortCircuit:
                break

        toPixPos0 = wcs0.skyToPixel(sky0)
        toPixPos1 = wcs1.skyToPixel(sky0)
        diffPix = math.hypot(*(toPixPos0 - toPixPos1))
        if diffPix > measDiffPix[0]:
            measDiffPix = (diffPix, sky0)
            if doShortCircuit:
                break

    msgList = []
    if measDiffSky[0] > maxDiffSky:
        msgList.append("%s arcsec max measured sky error > %s arcsec max allowed sky error at pix pos=%s" %
                       (measDiffSky[0].asArcseconds(), maxDiffSky.asArcseconds(), measDiffSky[1]))
    if measDiffPix[0] > maxDiffPix:
        msgList.append("%s max measured pix error > %s max allowed pix error at sky pos=%s" %
                       (measDiffPix[0], maxDiffPix, measDiffPix[1]))

    return "; ".join(msgList)


def wcsAlmostEqualOverBBox(wcs0, wcs1, bbox, maxDiffSky=0.01*arcseconds,
                           maxDiffPix=0.01, nx=5, ny=5):
    """Test if two :py:class:`WCS <lsst.afw.geom.SkyWcs>` are almost equal over a grid of pixel positions.

    Parameters
    ----------
    wcs0 : `lsst.afw.geom.SkyWcs`
        WCS 0
    wcs1 : `lsst.afw.geom.SkyWcs`
        WCS 1
    bbox : `lsst.afw.geom.Box2I` or `lsst.afw.geom.Box2D`
        boundaries of pixel grid over which to compare the WCSs
    maxDiffSky : `lsst.afw.geom.Angle`
        maximum separation between sky positions computed using Wcs.pixelToSky
    maxDiffPix : `float`
        maximum separation between pixel positions computed using Wcs.skyToPixel
    nx : `int`
        number of points in x for the grid of pixel positions
    ny : `int`
        number of points in y for the grid of pixel positions

    Returns
    -------
    almostEqual: `bool`
        `True` if two WCS are almost equal over a grid of pixel positions, else `False`
    """
    return not bool(_compareWcsOverBBox(
        wcs0=wcs0,
        wcs1=wcs1,
        bbox=bbox,
        maxDiffSky=maxDiffSky,
        maxDiffPix=maxDiffPix,
        nx=nx,
        ny=ny,
        doShortCircuit=True,
    ))


@lsst.utils.tests.inTestCase
def assertWcsAlmostEqualOverBBox(testCase, wcs0, wcs1, bbox, maxDiffSky=0.01*arcseconds,
                                 maxDiffPix=0.01, nx=5, ny=5, msg="WCSs differ"):
    """Assert that two :py:class:`WCS <lsst.afw.geom.SkyWcs>` are almost equal over a grid of pixel positions

    Compare pixelToSky and skyToPixel for two WCS over a rectangular grid of pixel positions.
    If the WCS are too divergent at any point, call testCase.fail; the message describes
    the largest error measured in pixel coordinates (if sky to pixel error was excessive)
    and sky coordinates (if pixel to sky error was excessive) across the entire pixel grid.

    Parameters
    ----------
    testCase : `unittest.TestCase`
        test case the test is part of; an object supporting one method: fail(self, msgStr)
    wcs0 : `lsst.afw.geom.SkyWcs`
        WCS 0
    wcs1 : `lsst.afw.geom.SkyWcs`
        WCS 1
    bbox : `lsst.afw.geom.Box2I` or `lsst.afw.geom.Box2D`
        boundaries of pixel grid over which to compare the WCSs
    maxDiffSky : `lsst.afw.geom.Angle`
        maximum separation between sky positions computed using Wcs.pixelToSky
    maxDiffPix : `float`
        maximum separation between pixel positions computed using Wcs.skyToPixel
    nx : `int`
        number of points in x for the grid of pixel positions
    ny : `int`
        number of points in y for the grid of pixel positions
    msg : `str`
        exception message prefix; details of the error are appended after ": "
    """
    errMsg = _compareWcsOverBBox(
        wcs0=wcs0,
        wcs1=wcs1,
        bbox=bbox,
        maxDiffSky=maxDiffSky,
        maxDiffPix=maxDiffPix,
        nx=nx,
        ny=ny,
        doShortCircuit=False,
    )
    if errMsg:
        testCase.fail("%s: %s" % (msg, errMsg))


@lsst.utils.tests.inTestCase
def assertWcsNearlyEqualOverBBox(*args, **kwargs):
    warnings.warn("Deprecated. Use assertWcsAlmostEqualOverBBox",
                  DeprecationWarning, 2)
    assertWcsAlmostEqualOverBBox(*args, **kwargs)


@lsst.utils.tests.inTestCase
def makeEndpoints(testCase):
    """Generate a representative sample of ``Endpoints``.

    Parameters
    ----------
    testCase : `unittest.TestCase`
        test case the test is part of; an object supporting one method: fail(self, msgStr)

    Returns
    -------
    endpoints : `list`
        List of endpoints with enough diversity to exercise ``Endpoint``-related
        code. Each invocation of this method shall return independent objects.
    """
    return [GenericEndpoint(n) for n in range(1, 6)] + \
           [Point2Endpoint(), SpherePointEndpoint()]


@lsst.utils.tests.inTestCase
def assertAnglesNearlyEqual(*args, **kwargs):
    warnings.warn("Deprecated. Use assertAnglesAlmostEqual",
                  DeprecationWarning, 2)
    assertAnglesAlmostEqual(*args, **kwargs)


@lsst.utils.tests.inTestCase
def assertPairsNearlyEqual(*args, **kwargs):
    warnings.warn("Deprecated. Use assertPairsAlmostEqual", DeprecationWarning, 2)
    assertPairsAlmostEqual(*args, **kwargs)


@lsst.utils.tests.inTestCase
def assertBoxesNearlyEqual(*args, **kwargs):
    warnings.warn("Deprecated. Use assertBoxesAlmostEqual", DeprecationWarning, 2)
    assertBoxesAlmostEqual(*args, **kwargs)
