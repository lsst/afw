#
# LSST Data Management System
# Copyright 2008-2017 LSST/AURA.
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
from __future__ import absolute_import, division, print_function

__all__ = ["makeImageFromArray", "makeMaskFromArray", "makeMaskedImageFromArrays",
           "wcsNearlyEqualOverBBox", "assertWcsNearlyEqualOverBBox"]

from builtins import str
import itertools
import math

import numpy

import lsst.utils.tests
import lsst.afw.geom as afwGeom
from .image import Image
from .mask import Mask
from .maskedImage import MaskedImage


def makeImageFromArray(array):
    """Construct an Image from a NumPy array, inferring the Image type from
    the NumPy type. Return None if input is None.
    """
    if array is None:
        return None
    return Image(array, dtype=array.dtype.type)


def makeMaskFromArray(array):
    """Construct an Mask from a NumPy array, inferring the Mask type from the
    NumPy type. Return None if input is None.
    """
    if array is None:
        return None
    return Mask(array, dtype=array.dtype.type)


def makeMaskedImageFromArrays(image, mask=None, variance=None):
    """Construct a MaskedImage from three NumPy arrays, inferring the
    MaskedImage types from the NumPy types.
    """
    return MaskedImage(makeImageFromArray(image), makeMaskFromArray(mask),
                       makeImageFromArray(variance), dtype=image.dtype.type)


def _compareWcsOverBBox(wcs0, wcs1, bbox, maxDiffSky=0.01*afwGeom.arcseconds,
                        maxDiffPix=0.01, nx=5, ny=5, doShortCircuit=True):
    """!Compare two WCS over a rectangular grid of pixel positions

    @param[in] wcs0  WCS 0 (an lsst.afw.image.Wcs)
    @param[in] wcs1  WCS 1 (an lsst.afw.image.Wcs)
    @param[in] bbox  boundaries of pixel grid over which to compare the WCSs (an lsst.afw.geom.Box2I or Box2D)
    @param[in] maxDiffSky  maximum separation between sky positions computed using Wcs.pixelToSky
        (an lsst.afw.geom.Angle)
    @param[in] maxDiffPix  maximum separation between pixel positions computed using Wcs.skyToPixel
    @param[in] nx  number of points in x for the grid of pixel positions
    @param[in] ny  number of points in y for the grid of pixel positions
    @param[in] doShortCircuit  if True then stop at the first error, else test all values in the grid
        and return information about the worst violations found

    @return return an empty string if the WCS are sufficiently close; else return a string describing
    the largest error measured in pixel coordinates (if sky to pixel error was excessive) and sky coordinates
    (if pixel to sky error was excessive). If doShortCircuit is true then the reported error is likely to be
    much less than the maximum error across the whole pixel grid.
    """
    if nx < 1 or ny < 1:
        raise RuntimeError("nx = %s and ny = %s must both be positive" % (nx, ny))
    if maxDiffSky <= 0*afwGeom.arcseconds:
        raise RuntimeError("maxDiffSky = %s must be positive" % (maxDiffSky,))
    if maxDiffPix <= 0:
        raise RuntimeError("maxDiffPix = %s must be positive" % (maxDiffPix,))

    bboxd = afwGeom.Box2D(bbox)
    xList = numpy.linspace(bboxd.getMinX(), bboxd.getMaxX(), nx)
    yList = numpy.linspace(bboxd.getMinY(), bboxd.getMaxY(), ny)
    # we don't care about measured error unless it is too large, so initialize to max allowed
    measDiffSky = (maxDiffSky, "?") # (sky diff, pix pos)
    measDiffPix = (maxDiffPix, "?") # (pix diff, sky pos)
    for x, y in itertools.product(xList, yList):
        fromPixPos = afwGeom.Point2D(x, y)
        sky0 = wcs0.pixelToSky(fromPixPos)
        sky1 = wcs1.pixelToSky(fromPixPos)
        diffSky = sky0.angularSeparation(sky1)
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

def wcsNearlyEqualOverBBox(wcs0, wcs1, bbox, maxDiffSky=0.01*afwGeom.arcseconds,
    maxDiffPix=0.01, nx=5, ny=5):
    """!Return True if two WCS are nearly equal over a grid of pixel positions, else False

    @param[in] wcs0  WCS 0 (an lsst.afw.image.Wcs)
    @param[in] wcs1  WCS 1 (an lsst.afw.image.Wcs)
    @param[in] bbox  boundaries of pixel grid over which to compare the WCSs (an lsst.afw.geom.Box2I or Box2D)
    @param[in] maxDiffSky  maximum separation between sky positions computed using Wcs.pixelToSky
        (an lsst.afw.geom.Angle)
    @param[in] maxDiffPix  maximum separation between pixel positions computed using Wcs.skyToPixel
    @param[in] nx  number of points in x for the grid of pixel positions
    @param[in] ny  number of points in y for the grid of pixel positions
    """
    return not bool(_compareWcsOverBBox(
        wcs0 = wcs0,
        wcs1 = wcs1,
        bbox = bbox,
        maxDiffSky = maxDiffSky,
        maxDiffPix = maxDiffPix,
        nx = nx,
        ny = ny,
        doShortCircuit = True,
    ))

@lsst.utils.tests.inTestCase
def assertWcsNearlyEqualOverBBox(testCase, wcs0, wcs1, bbox, maxDiffSky=0.01*afwGeom.arcseconds,
    maxDiffPix=0.01, nx=5, ny=5, msg="WCSs differ"):
    """!Compare pixelToSky and skyToPixel for two WCS over a rectangular grid of pixel positions

    If the WCS are too divergent, call testCase.fail; the message describes the largest error measured
    in pixel coordinates (if sky to pixel error was excessive) and sky coordinates (if pixel to sky error
    was excessive) across the entire pixel grid.

    @param[in] testCase  unittest.TestCase instance the test is part of;
                        an object supporting one method: fail(self, msgStr)
    @param[in] wcs0  WCS 0 (an lsst.afw.image.Wcs)
    @param[in] wcs1  WCS 1 (an lsst.afw.image.Wcs)
    @param[in] bbox  boundaries of pixel grid over which to compare the WCSs (an lsst.afw.geom.Box2I or Box2D)
    @param[in] maxDiffSky  maximum separation between sky positions computed using Wcs.pixelToSky
        (an lsst.afw.geom.Angle)
    @param[in] maxDiffPix  maximum separation between pixel positions computed using Wcs.skyToPixel
    @param[in] nx  number of points in x for the grid of pixel positions
    @param[in] ny  number of points in y for the grid of pixel positions
    @param[in] msg  exception message prefix; details of the error are appended after ": "
    """
    errMsg = _compareWcsOverBBox(
        wcs0 = wcs0,
        wcs1 = wcs1,
        bbox = bbox,
        maxDiffSky = maxDiffSky,
        maxDiffPix = maxDiffPix,
        nx = nx,
        ny = ny,
        doShortCircuit = False,
    )
    if errMsg:
        testCase.fail("%s: %s" % (msg, errMsg))
