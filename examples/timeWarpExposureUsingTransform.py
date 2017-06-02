#!/usr/bin/env python

#
# LSST Data Management System
# Copyright 2017 LSST Corporation.
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
from builtins import range
import sys
import os
import time

import lsst.utils
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath

MaxIter = 20
MaxTime = 1.0  # seconds
WarpSubregion = True  # set False to warp more pixels
SaveImages = False

afwdataDir = lsst.utils.getPackageDir("afwdata")

InputExposurePath = os.path.join(
    afwdataDir, "ImSim/calexp/v85408556-fr/R23/S11.fits")


def timeWarp(destMaskedImage, srcMaskedImage, destToSrc, warpingControl):
    """Time warpImage

    Parameters
    ----------
    destMaskedImage : `lsst.afw.image.MaskedImage`
        Destination (output) masked image
    srcMaskedImage : `lsst.afw.image.MaskedImage`
        Source (input) masked image
    destToSrc : `lsst.afw.geom.TransformPoint2ToPoint2`
        Transform from source pixels to destination pixels
    warpingControl : `lsst.afw.geom.WarpingControl`
        Warning control parameters

    Returns
    -------
    `tuple(float, int, int)`
        - Elapsed time in seconds
        - Number of iterations
        - Number of good pixels
    """
    startTime = time.time()
    for nIter in range(1, MaxIter + 1):
        goodPix = afwMath.warpImage(destMaskedImage, srcMaskedImage, destToSrc, warpingControl)
        endTime = time.time()
        if endTime - startTime > MaxTime:
            break

    return (endTime - startTime, nIter, goodPix)


def run():
    if len(sys.argv) < 2:
        srcExposure = afwImage.ExposureF(InputExposurePath)
        if WarpSubregion:
            bbox = afwGeom.Box2I(afwGeom.Point2I(
                0, 0), afwGeom.Extent2I(2000, 2000))
            srcExposure = afwImage.ExposureF(
                srcExposure, bbox, afwImage.LOCAL, False)
    else:
        srcExposure = afwImage.ExposureF(sys.argv[1])
    srcMaskedImage = srcExposure.getMaskedImage()
    srcMetadata = afwImage.DecoratedImageF(InputExposurePath).getMetadata()
    srcWcs = afwGeom.SkyWcs(srcMetadata)
    srcDim = srcExposure.getDimensions()
    srcCtrPos = afwGeom.Box2D(srcMaskedImage.getBBox()).getCenter()
    srcCtrSky = srcWcs.applyForward(srcCtrPos)
    srcScale = srcWcs.getPixelScale()

    # make the destination exposure small enough that even after rotation and offset
    # (by reasonable amounts) there are no edge pixels
    destDim = afwGeom.Extent2I(*[int(sd * 0.5) for sd in srcDim])
    destMaskedImage = afwImage.MaskedImageF(destDim)
    destCrPix = afwGeom.Box2D(destMaskedImage.getBBox()).getCenter()

    maskKernelName = ""
    cacheSize = 0

    print("Warping", InputExposurePath)
    print("Source (sub)image size:", srcDim)
    print("Destination image size:", destDim)
    print()

    print("test# interp  scaleFac    destSkyOff    rotAng   kernel   goodPix time/iter")
    print('       (pix)              (bear°, len")    (deg)                      (sec)')
    testNum = 1
    for interpLength in (0, 1, 5, 10):
        for scaleFac in (1.2,):
            destScale = srcScale / scaleFac
            for offsetOrientDegLenArcsec in ((0.0, 0.0),):  # ((0.0, 0.0), (-35.0, 10.5)):
                # offset (bearing, length) from sky at center of source to sky at center of dest
                offset = (offsetOrientDegLenArcsec[0] * afwGeom.degrees,
                          offsetOrientDegLenArcsec[1] * afwGeom.arcseconds)
                destCtrSky = srcCtrSky.offset(*offset)
                for rotAngDeg, kernelName in (
                    (0.0, "bilinear"),
                    (0.0, "lanczos2"),
                    (0.0, "lanczos3"),
                    (45.0, "lanczos3"),
                ):
                    warpingControl = afwMath.WarpingControl(
                        kernelName,
                        maskKernelName,
                        cacheSize,
                        interpLength,
                    )
                    destWcs = afwGeom.SkyWcs(
                        crpix = destCrPix,
                        crval = destCtrSky,
                        cdMatrix = afwGeom.makeCdMatrix(scale = destScale,
                                                        orientation = rotAngDeg * afwGeom.degrees,
                                                        flipX = False))
                    destToSrc = destWcs.then(srcWcs.getInverse())
                    dTime, nIter, goodPix = timeWarp(
                        destMaskedImage, srcMaskedImage, destToSrc, warpingControl)
                    print("%4d  %5d  %8.1f  %6.1f, %6.1f  %7.1f %10s %8d %6.2f" % (
                        testNum, interpLength, scaleFac, offsetOrientDegLenArcsec[0],
                        offsetOrientDegLenArcsec[1], rotAngDeg, kernelName, goodPix, dTime/float(nIter)))

                    if SaveImages:
                        destMaskedImage.writeFits(
                            "warpedMaskedImage%03d.fits" % (testNum,))
                    testNum += 1


if __name__ == "__main__":
    run()
