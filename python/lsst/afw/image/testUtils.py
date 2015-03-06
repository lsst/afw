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

##\file
## \brief Utilities to help write tests, mostly using numpy 
import math

import numpy

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom

def makeGaussianNoiseMaskedImage(dimensions, sigma, variance=1.0):
    """Make a gaussian noise MaskedImageF
    
    Inputs:
    - dimensions: dimensions of output array (cols, rows)
    - sigma; sigma of image plane's noise distribution
    - variance: constant value for variance plane
    """
    npSize = (dimensions[1], dimensions[0])
    image = numpy.random.normal(loc=0.0, scale=sigma, size=npSize).astype(numpy.float32)
    mask = numpy.zeros(npSize, dtype=numpy.uint16)
    variance = numpy.zeros(npSize, dtype=numpy.float32) + variance
    
    return afwImage.makeMaskedImageFromArrays(image, mask, variance)

def imagesDiffer(imageArr1, imageArr2, skipMaskArr=None, rtol=1.0e-05, atol=1e-08):
    """Compare the pixels of two image arrays; return True if close, False otherwise
    
    Inputs:
    - image1: first image to compare
    - image2: second image to compare
    - skipMaskArr: pixels to ignore; nonzero values are skipped
    - rtol: relative tolerance (see below)
    - atol: absolute tolerance (see below)
    
    rtol and atol are positive, typically very small numbers.
    The relative difference (rtol * abs(b)) and the absolute difference "atol" are added together
    to compare against the absolute difference between "a" and "b".
    
    Return a string describing the error if the images differ significantly, an empty string otherwise
    """
    retStrs = []
    if skipMaskArr is not None:
        maskedArr1 = numpy.ma.array(imageArr1, copy=False, mask = skipMaskArr)
        maskedArr2 = numpy.ma.array(imageArr2, copy=False, mask = skipMaskArr)
        filledArr1 = maskedArr1.filled(0.0)
        filledArr2 = maskedArr2.filled(0.0)
    else:
        filledArr1 = imageArr1
        filledArr2 = imageArr2

    nan1 = numpy.isnan(filledArr1)
    nan2 = numpy.isnan(filledArr2)
    if numpy.any(nan1 != nan2):
        retStrs.append("NaNs differ")

    posinf1 = numpy.isposinf(filledArr1)
    posinf2 = numpy.isposinf(filledArr2)
    if numpy.any(posinf1 != posinf2):
        retStrs.append("+infs differ")

    neginf1 = numpy.isneginf(filledArr1)
    neginf2 = numpy.isneginf(filledArr2)
    if numpy.any(neginf1 != neginf2):
        retStrs.append("-infs differ")

    # compare values that should be comparable (are neither infinite, nan nor masked)
    valSkipMaskArr = nan1 | nan2 | posinf1 | posinf2 | neginf1 | neginf2
    if skipMaskArr is not None:
        valSkipMaskArr |= skipMaskArr
    valMaskedArr1 = numpy.ma.array(imageArr1, copy=False, mask = valSkipMaskArr)
    valMaskedArr2 = numpy.ma.array(imageArr2, copy=False, mask = valSkipMaskArr)
    valFilledArr1 = valMaskedArr1.filled(0.0)
    valFilledArr2 = valMaskedArr2.filled(0.0)
    
    if not numpy.allclose(valFilledArr1, valFilledArr2, rtol=rtol, atol=atol):
        errArr = numpy.abs(valFilledArr1 - valFilledArr2)
        maxErr = errArr.max()
        maxPosInd = numpy.where(errArr==maxErr)
        maxPosTuple = (maxPosInd[1][0], maxPosInd[0][0])
        errStr = "maxDiff=%s at position %s; value=%s vs. %s" % \
            (maxErr, maxPosTuple, valFilledArr1[maxPosInd][0], valFilledArr2[maxPosInd][0])
        retStrs.insert(0, errStr)
    return "; ".join(retStrs)

def masksDiffer(maskArr1, maskArr2, skipMaskArr=None):
    """Compare the pixels of two mask arrays; return True if they match, False otherwise
    
    Inputs:
    - mask1: first image to compare
    - mask2: second image to compare
    - skipMaskArr: pixels to ignore; nonzero values are skipped
    
    Return a string describing the error if the images differ significantly, an empty string otherwise
    """
    retStr = ""
    if skipMaskArr is not None:
        maskedArr1 = numpy.ma.array(maskArr1, copy=False, mask = skipMaskArr)
        maskedArr2 = numpy.ma.array(maskArr2, copy=False, mask = skipMaskArr)
        filledArr1 = maskedArr1.filled(0.0)
        filledArr2 = maskedArr2.filled(0.0)
    else:
        filledArr1 = maskArr1
        filledArr2 = maskArr2

    if numpy.any(filledArr1 != filledArr2):
        errArr = numpy.abs(filledArr1 - filledArr2)
        maxErr = errArr.max()
        maxPosInd = numpy.where(errArr==maxErr)
        maxPosTuple = (maxPosInd[1][0], maxPosInd[0][0])
        retStr = "maxDiff=%s at position %s; value=%s vs. %s" % \
            (maxErr, maxPosTuple, filledArr1[maxPosInd][0], filledArr2[maxPosInd][0])
        retStr = "masks differ"
    return retStr

def maskedImagesDiffer(maskedImageArrSet1, maskedImageArrSet2,
    doImage=True, doMask=True, doVariance=True, skipMaskArr=None, rtol=1.0e-05, atol=1e-08):
    """Compare pixels from two masked images
    
    Inputs:
    - maskedImageArrSet1: first masked image to compare as (image, mask, variance) arrays
    - maskedImageArrSet2: second masked image to compare as (image, mask, variance) arrays
    - doImage: compare image planes if True
    - doMask: compare mask planes if True
    - doVariance: compare variance planes if True
    - skipMaskArr: pixels to ingore on the image, mask and variance arrays; nonzero values are skipped
    - rtol: relative tolerance (see below)
    - atol: absolute tolerance (see below)
    
    rtol and atol are positive, typically very small numbers.
    The relative difference (rtol * abs(b)) and the absolute difference "atol" are added together
    to compare against the absolute difference between "a" and "b".
    
    Return a string describing the error if the images differ significantly, an empty string otherwise
    """
    retStrs = []
    for ind, (doPlane, planeName) in enumerate(((doImage, "image"),
                                                (doMask, "mask"),
                                                (doVariance, "variance"))):
        if not doPlane:
            continue

        if planeName == "mask":
            errStr = masksDiffer(maskedImageArrSet1[ind], maskedImageArrSet2[ind], skipMaskArr=skipMaskArr)
            if errStr:
                retStrs.append(errStr)
        else:
            errStr = imagesDiffer(maskedImageArrSet1[ind], maskedImageArrSet2[ind],
                skipMaskArr=skipMaskArr, rtol=rtol, atol=atol)
            if errStr:
                retStrs.append("%s planes differ: %s" % (planeName, errStr))
    return " | ".join(retStrs)

def assertWcssAlmostEqual(wcs0, wcs1, bbox, maxSkyErr=0.001*afwGeom.arcseconds, maxPixErr=0.001,
    nx=5, ny=5, msg="WCSs differ"):
    """Assert that two WCSs give nearly equal results for pixelToSky and skyToPixel

    @param[in] wcs0  WCS 0 (an lsst.afw.image.Wcs)
    @param[in] wcs1  WCS 1 (an lsst.afw.image.Wcs)
    @param[in] bbox  boundaries of pixel grid over which to compare the WCSs (an lsst.afw.geom.Box2I or Box2D)
    @param[in] maxSkyErr  maximum separation between sky positions computed using Wcs.pixelToSky
        (an lsst.afw.geom.Angle)
    @param[in] maxPixErr  maximum separation between pixel positions computed using Wcs.skyToPixel
    @param[in] nx  number of points in x for the grid of pixel positions
    @param[in] ny  number of points in y for the grid of pixel positions
    @param[in] msg  prefix for error messages; details of the error are appended after ": "

    @throw AssertionError if the two WCSs do not match sufficiently closely
    """
    bboxd = afwGeom.Box2D(bbox)
    xList = numpy.linspace(bboxd.getMinX(), bboxd.getMaxX(), nx)
    yList = numpy.linspace(bboxd.getMinY(), bboxd.getMaxY(), ny)
    for x in xList:
        for y in yList:
            fromPixPos = afwGeom.Point2D(x, y)
            sky0 = wcs0.pixelToSky(fromPixPos)
            sky1 = wcs1.pixelToSky(fromPixPos)
            skyErr = sky0.angularSeparation(sky1)
            if skyErr > maxSkyErr:
                raise AssertionError("%s: %s arcsec sky error > %f arcsec max sky error at pixPos=%s" %
                    (msg, skyErr.asArcseconds(), maxSkyErr.asArcseconds(), fromPixPos,))

            toPixPos0 = wcs0.skyToPixel(sky0)
            toPixPos1 = wcs1.skyToPixel(sky0)
            pixErr = math.hypot(*(toPixPos0 - toPixPos1))
            if pixErr > maxPixErr:
                raise AssertionError("%s: %f pix error > %f max pix error at skyPos=%s" %
                        (msg, pixErr, maxPixErr, sky0))
