from __future__ import absolute_import, division
# 
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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

"""Application Framework image-related classes including Image, Mask and MaskedImage
"""
import math

import numpy

import lsst.utils.tests
import lsst.afw.geom as afwGeom
from . import imageLib

__all__ = ["makeImageFromArray", "makeMaskFromArray", "makeMaskedImageFromArrays",
    "assertWcsNearlyEqualOverBBox"]

suffixes = {str(numpy.uint16): "U", str(numpy.int32): "I", str(numpy.float32): "F", str(numpy.float64): "D"}

def makeImageFromArray(array):
    """Construct an Image from a NumPy array, inferring the Image type from the NumPy type.
    Return None if input is None.
    """
    if array is None: return None
    cls = getattr(imageLib, "Image%s" % (suffixes[str(array.dtype.type)],))
    return cls(array)

def makeMaskFromArray(array):
    """Construct an Mask from a NumPy array, inferring the Mask type from the NumPy type.
    Return None if input is None.
    """
    if array is None: return None
    cls = getattr(imageLib, "Mask%s" % (suffixes[str(array.dtype.type)],))
    return cls(array)

def makeMaskedImageFromArrays(image, mask=None, variance=None):
    """Construct a MaskedImage from three NumPy arrays, inferring the MaskedImage types from the NumPy types.
    """
    cls = getattr(imageLib, "MaskedImage%s" % (suffixes[str(image.dtype.type)],))
    return cls(makeImageFromArray(image), makeMaskFromArray(mask), makeImageFromArray(variance))

@lsst.utils.tests.inTestCase
def assertWcsNearlyEqualOverBBox(testCase, wcs0, wcs1, bbox, maxDiffSky=0.01*afwGeom.arcseconds, maxDiffPix=0.01,
    nx=5, ny=5, msg="WCSs differ"):
    """Compare pixelToSky and skyToPixel for two WCS over a rectangular grid of pixel positions

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

    @throw AssertionError if the two WCSs do not match sufficiently closely
    """
    if nx < 1 or ny < 1:
        raise RuntimeError("nx = %s and ny = %s must both be positive" % (nx, ny))

    bboxd = afwGeom.Box2D(bbox)
    xList = numpy.linspace(bboxd.getMinX(), bboxd.getMaxX(), nx)
    yList = numpy.linspace(bboxd.getMinY(), bboxd.getMaxY(), ny)
    measDiffSky = (afwGeom.Angle(0), "?") # (sky diff, pix pos)
    measDiffPix = (0, "?") # (pix diff, sky pos)
    for x in xList:
        for y in yList:
            fromPixPos = afwGeom.Point2D(x, y)
            sky0 = wcs0.pixelToSky(fromPixPos)
            sky1 = wcs1.pixelToSky(fromPixPos)
            diffSky = sky0.angularSeparation(sky1)
            if diffSky > measDiffSky[0]:
                measDiffSky = (diffSky, fromPixPos)

            toPixPos0 = wcs0.skyToPixel(sky0)
            toPixPos1 = wcs1.skyToPixel(sky0)
            diffPix = math.hypot(*(toPixPos0 - toPixPos1))
            if diffPix > measDiffPix[0]:
                measDiffPix = (diffPix, sky0)

    msgList = []
    if measDiffSky[0] > maxDiffSky:
        msgList.append("%s arcsec max measured sky error > %s arcsec max allowed sky error at pix pos=%s" %
            (measDiffSky[0].asArcseconds(), maxDiffSky.asArcseconds(), measDiffSky[1]))
    if measDiffPix[0] > maxDiffPix:
        msgList.append("%s max measured pix error > %s max allowed pix error at sky pos=%s" %
                (measDiffPix[0], maxDiffPix, measDiffPix[1]))
    if msgList:
        testCase.fail("%s: %s" % (msg, "; ".join(msgList)))
