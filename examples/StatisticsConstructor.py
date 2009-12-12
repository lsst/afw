#!/usr/bin/env python
#
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage

# This code was written as a result of ticket #1090 to
# demonstrate how to call the Statistics Constructor directly.

def main():
    
    gaussFunction = afwMath.GaussianFunction2D(3, 2, 0.5)
    gaussKernel   = afwMath.AnalyticKernel(10, 10, gaussFunction)
    mimg          = afwImage.MaskedImageF(100, 100)
    mimValue      = (2, 0x0, 1)
    mimg.set(mimValue)

    # call with the factory function ... should get stats on the image plane
    fmt = "%-36s %-16s %3.1f\n"
    print fmt % ("Using makeStatistics:",  "(should be " + str(mimValue[0]) + ")",
                 afwMath.makeStatistics(mimg, afwMath.MEAN).getValue()),
    
    # call the constructor directly ... once for image plane, then for variance
    print fmt % ("Using Statistics on getImage():", "(should be " + str(mimValue[0]) + ")",
                 afwMath.StatisticsF(mimg.getImage(), mimg.getMask(), afwMath.MEAN).getValue()),
    print fmt % ("Using Statistics on getVariance():", "(should be " + str(mimValue[2]) + ")",
                 afwMath.StatisticsF(mimg.getVariance(), mimg.getMask(), afwMath.MEAN).getValue()),

    
if __name__ == '__main__':
    main()
