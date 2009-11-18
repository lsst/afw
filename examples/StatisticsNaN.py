#!/usr/bin/env python
#
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.afw.display.ds9 as ds9

# This code was submitted as a part of ticket #749 to demonstrate
# the failure of Statistics in dealing with NaN

disp = False

def main():
    
    gaussFunction = afwMath.GaussianFunction2D(3, 2, 0.5)
    gaussKernel   = afwMath.AnalyticKernel(10, 10, gaussFunction)
    inImage       = afwImage.ImageF(100, 100)
    inImage.set(1)
    if disp: ds9.mtv(inImage, frame = 0)
    
    # works
    outImage      = afwImage.ImageF(100, 100)
    afwMath.convolve(outImage, inImage, gaussKernel, False, True)
    if disp: ds9.mtv(outImage, frame = 1)
    print "Should be a number: ", afwMath.makeStatistics(outImage, afwMath.MEAN).getValue()
    print "Should be a number: ", afwMath.makeStatistics(outImage, afwMath.STDEV).getValue()
    
    # not works ... now does work
    outImage      = afwImage.ImageF(100, 100)
    afwMath.convolve(outImage, inImage, gaussKernel, False, False)
    if disp: ds9.mtv(outImage, frame = 2)
    print "Should be a number: ", afwMath.makeStatistics(outImage, afwMath.MEAN).getValue()
    print "Should be a number: ", afwMath.makeStatistics(outImage, afwMath.STDEV).getValue()

    # This will print nan
    sctrl = afwMath.StatisticsControl()
    sctrl.setNanSafe(False)
    print ("Should be a nan (nanSafe set to False): " +
           str(afwMath.makeStatistics(outImage, afwMath.MEAN, sctrl).getValue()))

if __name__ == '__main__':
    main()
