#!/usr/bin/env python
import optparse
import os
import sys

import eups

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.daf.base as dafBase
import lsst.pex.logging

def main():
    DefDataDir = eups.productDir("afwdata") or ""
    
    DefOriginalExposurePath = os.path.join(DefDataDir, "871034p_1_MI")
    DefWCSExposurePath = os.path.join(DefDataDir, "871034p_1_MI")
    DefOutputExposurePath = "warpedExposure"
    DefKernel = "lanczos3"
    DefVerbosity = 5 # change to 0 once this all works to hide all messages
    
    usage = """usage: %%prog [options] [originalExposure [warpedWCSExposure [outputExposure]]]

    Computes outputExposure = originalExposure warped to match warpedWCSExposure's WCS and size

    Note:
    - exposure arguments are paths to Exposure fits files;
      they must NOT include the final _img.fits|_var.fits|_msk.fits
    - default originalExposure = %s
    - default warpedWCSExposure = %s
    - default outputExposure = %s
    """ % (DefOriginalExposurePath, DefWCSExposurePath, DefOutputExposurePath)
    
    parser = optparse.OptionParser(usage)
    parser.add_option("", "--kernel",
                      type=str, default=DefKernel,
                      help="kernel type: bilinear or lancszosN where N = order; default=%s" % (DefKernel,))
    parser.add_option("-v", "--verbosity",
                      type=int, default=DefVerbosity,
                      help="verbosity of diagnostic trace messages; 1 for just warnings, more for more" + \
                      " information; default=%s" % (DefVerbosity,))
    
    (opt, args) = parser.parse_args()
    
    kernelName = opt.kernel.lower()
    if kernelName == "bilinear":
        print "Bilinear"
        kernel = afwMath.BilinearWarpingKernel()
    elif kernelName.startswith("lanczos"):
        kernelOrder = int(kernelName[7:])
        print "Lanczos order", kernelOrder
        kernel = afwMath.LanczosWarpingKernel(kernelOrder)
    else:
        print "Error: unknown kernel %r" % (kernelName,)
        parser.help()
        sys.exit(1)
    
    def getArg(ind, defValue):
        if ind < len(args):
            return args[ind]
        return defValue
    
    originalExposurePath = getArg(0, DefWCSExposurePath)
    warpedWCSExposurePath = getArg(1, DefOriginalExposurePath)
    outputExposurePath = getArg(2, DefOutputExposurePath)
    print "Remapping masked image  ", originalExposurePath
    print "to match wcs and size of", warpedWCSExposurePath
    
    originalExposure = afwImage.ExposureD(originalExposurePath)
    print "originalExposure=%r" % (originalExposure,)
    
    warpedExposure = afwImage.ExposureD(warpedWCSExposurePath)
    print "warpedExposure=%r" % (warpedExposure,)
    
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
