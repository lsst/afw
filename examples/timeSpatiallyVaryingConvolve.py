#!/usr/bin/env python
import math
import os
import time
import random

import eups

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.math.detail as mathDetail

MaxIter = 20
MaxTime = 2.0 # seconds

# force the same seed each time for reproducible results;
# set this seed before computing each kernel that uses random numbers
RandomSeed = 1.0


dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("Must set up afwdata")

InputMaskedImagePath = os.path.join(dataDir, "med")

def getAnalyticKernel(kSize, imSize):
    """Return spatially varying analytic kernel
    
    @param kSize: kernel size (height = width)
    @param imSize: image size (x, y)
    
    @return a Gaussian analytic kernel with a 1st order polynomial spatial model
    """
    # construct analytic kernel
    gaussFunc = afwMath.GaussianFunction2D(1, 1, 0)
    polyOrder = 1;
    polyFunc = afwMath.PolynomialFunction2D(polyOrder)
    kernel = afwMath.AnalyticKernel(kSize, kSize, gaussFunc, polyFunc)
    
    minSigma = 0.1
    maxSigma = 3.0

    # get copy of spatial parameters (all zeros), set and feed back to the kernel
    polyParamsList = [[0.0]*3]*3
    polyParamsList[0][0] = minSigma
    polyParamsList[0][1] = (maxSigma - minSigma) / float(imSize[0])
    polyParamsList[0][2] = 0.0
    polyParamsList[1][0] = minSigma;
    polyParamsList[1][1] = 0.0;
    polyParamsList[1][2] = (maxSigma - minSigma) / float(imSize[1]);
    kernel.setSpatialParameters(polyParamsList);
    return kernel

def getDeltaLinearCombinationKernel(kSize, imSize):
    """Return a LinearCombinationKernel of delta functions

    @param kSize: kernel size (height = width)
    @param imSize: image size (x, y)
    
    @return LinearCombinationKernel with a delta function basis and a 1st order polynomial spatial model
    """
    kernelList = afwMath.KernelList()
    for ctrX in range(kSize):
        for ctrY in range(kSize):
            kernelList.append(afwMath.DeltaFunctionKernel(kSize, kSize, afwImage.PointI(ctrX, ctrY)))
            
    polyOrder = 1;
    polyFunc = afwMath.PolynomialFunction2D(polyOrder)
    kernel = afwMath.LinearCombinationKernel(kernelList, polyFunc)
    
    random.seed(RandomSeed)

    polyParamsList = []
    for ind in range(len(kernelList)):
        polyParamsList.append([
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0 / float(imSize[0]), 1.0 / float(imSize[0])),
            random.uniform(-1.0 / float(imSize[1]), 1.0 / float(imSize[1])),
        ])
    kernel.setSpatialParameters(polyParamsList);
    return kernel

def getGaussianLinearCombinationKernel(kSize, imSize):
    """Return a LinearCombinationKernel with 5 bases, each a Gaussian
    """
    kernelList = afwMath.KernelList()
    for fwhmX, fwhmY, angle in (
        (2.0, 2.0, 0.0),
        (0.5, 4.0, 0.0),
        (0.5, 4.0, math.pi / 4.0),
        (0.5, 4.0, math.pi / 2.0),
        (4.0, 4.0, 0.0),
    ):
        gaussFunc = afwMath.GaussianFunction2D(fwhmX, fwhmY, angle)
        kernelList.append(afwMath.AnalyticKernel(kSize, kSize, gaussFunc))

    polyOrder = 1;
    polyFunc = afwMath.PolynomialFunction2D(polyOrder)
    kernel = afwMath.LinearCombinationKernel(kernelList, polyFunc)
    
    random.seed(RandomSeed)

    polyParamsList = []
    for ind in range(len(kernelList)):
        polyParamsList.append([
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0 / float(imSize[0]), 1.0 / float(imSize[0])),
            random.uniform(-1.0 / float(imSize[1]), 1.0 / float(imSize[1])),
        ])
    kernel.setSpatialParameters(polyParamsList);
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
        mathDetail.convolveWithInterpolation(outImage, inImage, kernel, convControl)
#        afwMath.convolve(outImage, inImage, kernel, convControl)
        endTime = time.time()
        if endTime - startTime > MaxTime:
            break

    return (endTime - startTime, nIter)

def timeSet(outImage, inImage, kernelFunction, kernelDescr, convControl):
    imSize = inImage.getDimensions()
    for methodDescr, maxInterpolationError in (
        ("Interpolation", 1000.0),
        ("Brute Force", 0.0),
        ("Default Mix of Interpolation and Brute Force", 1.0e-5),
    ):
        convControl.setMaxInterpolationError(maxInterpolationError)
        print "%s using %s" % (kernelDescr, methodDescr)
        print "ImWid\tImHt\tKerWid\tKerHt\tSec/Cnv"
        for kSize in (5, 11, 19):
            kernel = kernelFunction(kSize, imSize)
            dur, nIter = timeConvolution(outImage, inImage, kernel, convControl)
            print "%d\t%d\t%d\t%d\t%0.2f" % (imSize[0], imSize[1], kSize, kSize, dur/float(nIter))
    

def run():
    convControl = afwMath.ConvolutionControl()
    convControl.setDoNormalize(True)
    fullInImage = afwImage.MaskedImageF(InputMaskedImagePath)
    imSize = (256, 256)
    bbox = afwImage.BBox(afwImage.PointI(0, 0), imSize[0], imSize[1])
    inImage = afwImage.MaskedImageF(fullInImage, bbox, False)
    outImage = afwImage.MaskedImageF(inImage.getDimensions())
    
    timeSet(outImage, inImage, getAnalyticKernel,
        "Analytic Kernel", convControl)
    timeSet(outImage, inImage, getDeltaLinearCombinationKernel,
        "LinearCombinationKernel with Delta Function Basis", convControl)
    timeSet(outImage, inImage, getGaussianLinearCombinationKernel,
        "LinearCombinationKernel with 5 Gaussian Basis Kernels", convControl)

if __name__ == "__main__":
    run()
