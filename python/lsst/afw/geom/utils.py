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
import lsst.geom
from .endpoint import GenericEndpoint, Point2Endpoint, SpherePointEndpoint


def _compareWcsOverBBox(wcs0, wcs1, bbox, maxDiffSky=0.01*lsst.geom.arcseconds,
                        maxDiffPix=0.01, nx=5, ny=5, doShortCircuit=True):
    """Compare two :py:class:`WCS <lsst.afw.geom.SkyWcs>` over a rectangular grid of pixel positions

    Parameters
    ----------
    wcs0 : `lsst.afw.geom.SkyWcs`
        WCS 0
    wcs1 : `lsst.afw.geom.SkyWcs`
        WCS 1
    bbox : `lsst.geom.Box2I` or `lsst.geom.Box2D`
        boundaries of pixel grid over which to compare the WCSs
    maxDiffSky : `lsst.geom.Angle`
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
    if maxDiffSky <= 0*lsst.geom.arcseconds:
        raise RuntimeError("maxDiffSky = %s must be positive" % (maxDiffSky,))
    if maxDiffPix <= 0:
        raise RuntimeError("maxDiffPix = %s must be positive" % (maxDiffPix,))

    bboxd = lsst.geom.Box2D(bbox)
    xList = np.linspace(bboxd.getMinX(), bboxd.getMaxX(), nx)
    yList = np.linspace(bboxd.getMinY(), bboxd.getMaxY(), ny)
    # we don't care about measured error unless it is too large, so initialize
    # to max allowed
    measDiffSky = (maxDiffSky, "?")  # (sky diff, pix pos)
    measDiffPix = (maxDiffPix, "?")  # (pix diff, sky pos)
    for x, y in itertools.product(xList, yList):
        fromPixPos = lsst.geom.Point2D(x, y)
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


def wcsAlmostEqualOverBBox(wcs0, wcs1, bbox, maxDiffSky=0.01*lsst.geom.arcseconds,
                           maxDiffPix=0.01, nx=5, ny=5):
    """Test if two :py:class:`WCS <lsst.afw.geom.SkyWcs>` are almost equal over a grid of pixel positions.

    Parameters
    ----------
    wcs0 : `lsst.afw.geom.SkyWcs`
        WCS 0
    wcs1 : `lsst.afw.geom.SkyWcs`
        WCS 1
    bbox : `lsst.geom.Box2I` or `lsst.geom.Box2D`
        boundaries of pixel grid over which to compare the WCSs
    maxDiffSky : `lsst.geom.Angle`
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
def assertWcsAlmostEqualOverBBox(testCase, wcs0, wcs1, bbox, maxDiffSky=0.01*lsst.geom.arcseconds,
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
    bbox : `lsst.geom.Box2I` or `lsst.geom.Box2D`
        boundaries of pixel grid over which to compare the WCSs
    maxDiffSky : `lsst.geom.Angle`
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
