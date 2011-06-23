#!/usr/bin/env python
"""Report how many of each bit plane is set in the mask

Usage:
./countMaskBits.py path-to-masked-image
"""
import sys
import os

import numpy

import lsst.afw.image as afwImage

BadPixelList = ["BAD", "SAT", "CR"]

def getMaskBitNameDict(mask):
    """Compute a dictionary of bit index: bit plane name
    
    @param[in] mask: an afwImage.MaskU
    
    @return maskBitNameDict: a dictionary of bit index: bit plane name
    """
    maskBitNameDict = dict()
    maskNameBitDict = mask.getMaskPlaneDict()
    for name, ind in maskNameBitDict.iteritems():
        maskBitNameDict[ind] = name
    return maskBitNameDict

def countInterp(maskedImage):
    """Count how many BAD, SAT or CR pixels are interpolated over and how many are not
        
    @param[in] maskedImage: an afwImage MaskedImage

    @return
    - numBad: number of pixels that are BAD or SAT
    - numInterp: number of pixels that are interpolated
    - numBadAndInterp: number of pixels that are bad and interpolated
    """
    interpMask = afwImage.MaskU.getPlaneBitMask("INTRP")
    badMask = afwImage.MaskU.getPlaneBitMask(BadPixelList)
    
    maskArr = maskedImage.getMask().getArray()
    isBadArr = maskArr & badMask > 0
    isInterpArr = maskArr & interpMask > 0
    numBad = numpy.sum(isBadArr)
    numInterp = numpy.sum(isInterpArr)
    numBadAndInterp = numpy.sum(isBadArr & isInterpArr)
    return numBad, numInterp, numBadAndInterp

if __name__ == "__main__":
    maskedImage = afwImage.MaskedImageF(sys.argv[1])
    
    mask = maskedImage.getMask()
    maskBitNameDict = getMaskBitNameDict(mask)
    maskArr = maskedImage.getMask().getArray()
    bitIndList = sorted(maskBitNameDict.keys())
    print "Bit Mask Plane Name    # Pixels"
    for bitInd in bitIndList:
        planeName = maskBitNameDict[bitInd]
        bitMask = 1 << bitInd
        count = numpy.sum(maskArr & bitMask > 0)
        print "%3d %-18s %d" % (bitInd, planeName, count)

    print
    print "Interpolation: \"bad\" pixels have any of these bits set: %s" % (BadPixelList,)
    numBad, numInterp, numBadAndInterp = countInterp(maskedImage)
    print "%d bad; %d interp; %d bad & interp; %d bad and not interp; %d good but interp" % \
        (numBad, numInterp, numBadAndInterp, numBad - numBadAndInterp, numInterp - numBadAndInterp)
