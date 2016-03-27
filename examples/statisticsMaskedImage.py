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

# This example demonstrates how to compute statistics on a masked image

def main():
    showSetAndMask()

def showSetAndMask():

    nX, nY        = 10, 10
    mimg          = afwImage.MaskedImageF(afwGeom.Extent2I(nX, nY))

    # decide what we believe to be 'bad' mask bits
    mask          = mimg.getMask()
    maskbitBad    = mask.getPlaneBitMask('BAD')
    maskbitSat    = mask.getPlaneBitMask('SAT')
    maskbits      = maskbitBad | maskbitSat

    # four possibilities ... none, bad, sat, bad+sat
    # we'll just set the pixel value to equal the mask value for simplicity
    values = [
        (0, 0x0, 0),
        (maskbitBad, maskbitBad, 0),
        (maskbitSat, maskbitSat, 0),
        (maskbits, maskbits, 0)
        ]
    for iY in range(nY):
        for iX in range(nX):
            mod = (iY*nX + iX) % len(values)
            mimg.set(iX, iY, values[mod])


    # create our statisticsControl object and change the andMask to reject
    # the bits we deem bad.            
    masks = [0x0, maskbitBad, maskbitSat, maskbits]    
    explanations = [
        "all values accepted ... avg of 0,1,2,3 = 1.5",
        "reject 'BAD'        ... avg of 0,2     = 1.0",
        "reject 'SAT'        ... avg of 0,1     = 0.5",
        "reject BAD | SAT    ... avg of 0       = 0.0"
        ]
    for i in range(len(masks)):
        sctrl = afwMath.StatisticsControl()
        sctrl.setAndMask(masks[i])
        stat = afwMath.makeStatistics(mimg, afwMath.MEAN, sctrl)
        answer = stat.getValue(afwMath.MEAN)
        print explanations[i], " (got %.1f)" % (answer)

        
    
if __name__ == '__main__':
    main()
