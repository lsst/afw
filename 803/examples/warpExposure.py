#!/usr/bin/env python
import optparse
import os
import sys

import eups

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.daf.base as dafBase
import lsst.pex.logging

dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("Must set up afwdata to run these tests")

def main():
    DefDataDir = eups.productDir("afwdata") or ""
    
    DefOriginalExposurePath = os.path.join(DefDataDir, "med")
    DefWcsImageOrExposurePath = os.path.join(DefDataDir, "medswarp1lanczos4.fits")
    DefOutputExposurePath = "warpedExposure"
    DefKernel = "lanczos4"
    DefVerbosity = 6 # change to 0 once this all works to hide all messages
    
    usage = """usage: %%prog [options] [originalExposure [warpedWcsImageOrExposure [outputExposure]]]

    Computes outputExposure = originalExposure warped to match warpedWcsExposure's WCS and size

    Note:
    - exposure arguments are paths to Exposure fits files;
      they must NOT include the final _img.fits|_var.fits|_msk.fits
      if warpedWcsImageOrExposure ends in .fits then it specifies an image
    - default originalExposure = %s
    - default warpedWcsImageOrExposure = %s
    - default outputExposure = %s
    """ % (DefOriginalExposurePath, DefWcsImageOrExposurePath, DefOutputExposurePath)
    
    parser = optparse.OptionParser(usage)
    parser.add_option("-k", "--kernel",
                      type=str, default=DefKernel,
                      help="kernel type: bilinear or lancszosN where N = order; default=%s" % (DefKernel,))
    parser.add_option("-v", "--verbosity",
                      type=int, default=DefVerbosity,
                      help="verbosity of diagnostic trace messages; 1 for just warnings, more for more" + \
                      " information; default=%s" % (DefVerbosity,))
    
    (opt, args) = parser.parse_args()
    
    kernelName = opt.kernel.lower()
    kernel = afwMath.makeWarpingKernel(kernelName)
    
    def getArg(ind, defValue):
        if ind < len(args):
            return args[ind]
        return defValue
    
    originalExposurePath = getArg(0, DefOriginalExposurePath)
    warpedWcsImageOrExposurePath = getArg(1, DefWcsImageOrExposurePath)
    outputExposurePath = getArg(2, DefOutputExposurePath)
    print "Remapping masked image  ", originalExposurePath
    print "to match wcs and size of", warpedWcsImageOrExposurePath
    
    originalExposure = afwImage.ExposureF(originalExposurePath)
    
    if warpedWcsImageOrExposurePath.lower().endswith(".fits"):
        # user specified an image, not an exposure
        warpedDI = afwImage.DecoratedImageF(warpedWcsImageOrExposurePath)
        warpedWcs = afwImage.Wcs(warpedDI.getMetadata())
        warpedMI = afwImage.MaskedImageF(warpedDI.getWidth(), warpedDI.getHeight())
        warpedExposure = afwImage.ExposureF(warpedMI, warpedWcs)
    else:
        warpedExposure = afwImage.ExposureF(warpedWcsImageOrExposurePath)
    
    if opt.verbosity > 0:
        print "Verbosity =", opt.verbosity
        lsst.pex.logging.Trace_setVerbosity("lsst.afw.math", opt.verbosity)
    
    numGoodPixels = afwMath.warpExposure(warpedExposure, originalExposure, kernel)
    print "Warped exposure has %s good pixels" % (numGoodPixels)
    
    print "Writing warped exposure to %s" % (outputExposurePath,)
    warpedExposure.writeFits(outputExposurePath)

if __name__ == "__main__":
    memId0 = dafBase.Citizen_getNextMemId()
    main()
    # check for memory leaks
    if dafBase.Citizen_census(0, memId0) != 0:
        print dafBase.Citizen_census(0, memId0), "Objects leaked:"
        print dafBase.Citizen_census(dafBase.cout, memId0)
