#!/usr/bin/env python

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

#
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom

# This code was written as a result of ticket #1090 to
# demonstrate how to call the Statistics Constructor directly.

def main():
    
    mimg          = afwImage.MaskedImageF(afwGeom.Extent2I(100, 100))
    mimValue      = (2, 0x0, 1)
    mimg.set(mimValue)

    # call with the factory function ... should get stats on the image plane
    fmt = "%-40s %-16s %3.1f\n"
    print fmt % ("Using makeStatistics:",  "(should be " + str(mimValue[0]) + ")",
                 afwMath.makeStatistics(mimg, afwMath.MEAN).getValue()),
    
    # call the constructor directly ... once for image plane, then for variance
    # - make sure we're not using weighted stats for this
    sctrl = afwMath.StatisticsControl()
    sctrl.setWeighted(False)
    print fmt % ("Using Statistics on getImage():", "(should be " + str(mimValue[0]) + ")",
                 afwMath.StatisticsF(mimg.getImage(), mimg.getMask(), mimg.getVariance(),
                                     afwMath.MEAN, sctrl).getValue()),
    print fmt % ("Using Statistics on getVariance():", "(should be " + str(mimValue[2]) + ")",
                 afwMath.StatisticsF(mimg.getVariance(), mimg.getMask(), mimg.getVariance(),
                                     afwMath.MEAN, sctrl).getValue()),

    # call makeStatistics as a front-end for the constructor
    print fmt % ("Using makeStatistics on getImage():", "(should be " + str(mimValue[0]) + ")",
                 afwMath.makeStatistics(mimg.getImage(), mimg.getMask(), afwMath.MEAN).getValue()),
    print fmt % ("Using makeStatistics on getVariance():", "(should be " + str(mimValue[2]) + ")",
                 afwMath.makeStatistics(mimg.getVariance(), mimg.getMask(), afwMath.MEAN).getValue()),

    
if __name__ == '__main__':
    main()
