#!/usr/bin/env python

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

from __future__ import print_function
from __future__ import division
from builtins import range
import math
import sys
import os
import time

import lsst.utils
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
from lsst.log import Log

Log.getDefaultLogger().setLevel(Log.INFO)
Log.getLogger("TRACE2.afw.math.convolve").setLevel(Log.DEBUG)

MaxIter = 20
MaxTime = 1.0 # seconds

afwdataDir = lsst.utils.getPackageDir("afwdata")

InputMaskedImagePath = os.path.join(afwdataDir, "data", "med.fits")

def getSpatialParameters(nKernelParams, func):
    """Get basic spatial parameters list

    You may wish to tweak it up for specific cases (especially the lower order terms)
    """
    nCoeff = func.getNParameters()
    spParams = [[0.0]*nCoeff]*nKernelParams
    for kernelTermInd in range(nKernelParams):
        spParams[kernelTermInd][0] = 1.0
        spParams[kernelTermInd][1:3] = [1.0e-3]*len(spParams[kernelTermInd][1:3])
        spParams[kernelTermInd][3:6] = [1.0e-6]*len(spParams[kernelTermInd][3:6])
        spParams[kernelTermInd][6:10] = [1.0e-9]*len(spParams[kernelTermInd][6:10])
    return spParams

def getAnalyticKernel(kSize, imSize, spOrder):
    """Return spatially varying analytic kernel: a Gaussian

    @param kSize: kernel size (scalar; height = width)
    @param x, y imSize: image size
    """
    gaussFunc = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
    polyFunc = afwMath.PolynomialFunction2D(spOrder)
    kernel = afwMath.AnalyticKernel(kSize, kSize, gaussFunc, polyFunc)

    minSigma = 0.1
    maxSigma = 3.0

    spParams = getSpatialParameters(3, polyFunc)
    spParams[0][0:3] = [minSigma, (maxSigma - minSigma) / float(imSize[0]), 0.0]
    spParams[1][0:3] = [minSigma, 0.0, (maxSigma - minSigma) / float(imSize[1])]
    kernel.setSpatialParameters(spParams);
    return kernel

def getSeparableKernel(kSize, imSize, spOrder):
    """Return spatially varying separable kernel: a pair of 1-d Gaussians

    @param kSize: kernel size (scalar; height = width)
    @param x, y imSize: image size
    """
    gaussFunc = afwMath.GaussianFunction1D(1)
    polyFunc = afwMath.PolynomialFunction2D(spOrder)
    kernel = afwMath.SeparableKernel(kSize, kSize, gaussFunc, gaussFunc, polyFunc)

    minSigma = 0.1
    maxSigma = 3.0

    spParams = getSpatialParameters(2, polyFunc)
    spParams[0][0:3] = [minSigma, (maxSigma - minSigma) / float(imSize[0]), 0.0]
    spParams[1][0:3] = [minSigma, 0.0, (maxSigma - minSigma) / float(imSize[0])]
    kernel.setSpatialParameters(spParams);
    return kernel

def getDeltaLinearCombinationKernel(kSize, imSize, spOrder):
    """Return a LinearCombinationKernel of delta functions

    @param kSize: kernel size (scalar; height = width)
    @param x, y imSize: image size
    """
    kernelList = []
    for ctrX in range(kSize):
        for ctrY in range(kSize):
            kernelList.append(afwMath.DeltaFunctionKernel(kSize, kSize, afwGeom.Point2I(ctrX, ctrY)))

    polyFunc = afwMath.PolynomialFunction2D(spOrder)
    kernel = afwMath.LinearCombinationKernel(kernelList, polyFunc)

    spParams = getSpatialParameters(len(kernelList), polyFunc)
    kernel.setSpatialParameters(spParams);
    return kernel

def getGaussianLinearCombinationKernel(kSize, imSize, spOrder):
    """Return a LinearCombinationKernel with 5 bases, each a Gaussian

    @param kSize: kernel size (scalar; height = width)
    @param x, y imSize: image size
    """
    kernelList = []
    for fwhmX, fwhmY, angle in (
        (2.0, 2.0, 0.0),
        (0.5, 4.0, 0.0),
        (0.5, 4.0, math.pi / 4.0),
        (0.5, 4.0, math.pi / 2.0),
        (4.0, 4.0, 0.0),
    ):
        gaussFunc = afwMath.GaussianFunction2D(fwhmX, fwhmY, angle)
        kernelList.append(afwMath.AnalyticKernel(kSize, kSize, gaussFunc))

    polyFunc = afwMath.PolynomialFunction2D(spOrder)
    kernel = afwMath.LinearCombinationKernel(kernelList, polyFunc)

    spParams = getSpatialParameters(len(kernelList), polyFunc)
    kernel.setSpatialParameters(spParams);
    return kernel


def timeConvolution(outImage, inImage, kernel, convControl):
    """Time convolution

    @param outImage: output image or masked image (must be the same size as inImage)
    @param inImage: input image or masked image
    @param kernel: convolution kernel
    @param convControl: convolution control parameters (afwMath.ConvolutionControl)

    @return (elapsed time in seconds, number of iterations)
    """
    startTime = time.time();
    for nIter in range(1, MaxIter + 1):
#        mathDetail.convolveWithInterpolation(outImage, inImage, kernel, convControl)
        afwMath.convolve(outImage, inImage, kernel, convControl)
        endTime = time.time()
        if endTime - startTime > MaxTime:
            break

    return (endTime - startTime, nIter)

def timeSet(outImage, inImage, kernelFunction, kernelDescr, convControl, spOrder, doInterp=True):
    """Time a set of convolutions for various parameters

    Inputs:
    ... the usual
    - spOrder: the order of the spatial Polynomial2 function
    - doInterp: if True then test interpolation, else only test brute force
    """
    imSize = inImage.getDimensions()
    if doInterp:
        methodDescrInterpDistList = (
            ("no interpolation", 0),
            ("linear interpolation over 10 x 10 pixels", 10),
            ("linear interpolation over 20 x 20 pixels", 20),
        )
    else:
        methodDescrInterpDistList = (
            ("no interpolation", 0),
        )
    for methodDescr, maxInterpolationDistance in methodDescrInterpDistList:
        convControl.setMaxInterpolationDistance(maxInterpolationDistance)
        print("%s using %s" % (kernelDescr, methodDescr))
        print("ImWid\tImHt\tKerWid\tKerHt\tSec/Cnv")
        for kSize in (5, 11, 19):
            kernel = kernelFunction(kSize, imSize, spOrder)
            dur, nIter = timeConvolution(outImage, inImage, kernel, convControl)
            print("%d\t%d\t%d\t%d\t%0.2f" % (imSize[0], imSize[1], kSize, kSize, dur/float(nIter)))
    print()

def run():
    convControl = afwMath.ConvolutionControl()
    convControl.setDoNormalize(True)
    spOrder = 3
    print("All kernels use a spatial model of a Polynomial2 of order %s" % (spOrder,))

    for imageClass in (
        afwImage.ImageF,
        afwImage.ImageD,
        afwImage.MaskedImageF,
        afwImage.MaskedImageD,
    ):
        print("\n*** Test convolution with %s ***\n" % (imageClass.__name__,))
        if len(sys.argv) < 2:
            inImage = imageClass(InputMaskedImagePath)
            # to get original behavior change True to False:
            if (False):
                bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(256, 256))
                inImage = imageClass(inImage, bbox, afwImage.LOCAL, False)
        else:
            inImage = imageClass(sys.argv[1])
        outImage = imageClass(inImage.getDimensions())

        timeSet(outImage, inImage, getAnalyticKernel,
            "AnalyticKernel", convControl, spOrder=spOrder)
        timeSet(outImage, inImage, getSeparableKernel,
            "SeparableKernel", convControl, spOrder=spOrder, doInterp=False)
        timeSet(outImage, inImage, getGaussianLinearCombinationKernel,
            "LinearCombinationKernel with 5 Gaussian Basis Kernels", convControl, spOrder=spOrder)
        timeSet(outImage, inImage, getDeltaLinearCombinationKernel,
            "LinearCombinationKernel with Delta Function Basis", convControl, spOrder=spOrder)

if __name__ == "__main__":
    run()
