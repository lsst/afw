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

import optparse
import os

import eups

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.daf.base as dafBase
import lsst.pex.logging

def main():
    DefDataDir = eups.productDir("afwdata") or ""
    
    DefOriginalExposurePath = os.path.join(DefDataDir, "data", "med")
    DefWcsImageOrExposurePath = os.path.join(DefDataDir, "data", "medswarp1lanczos4.fits")
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
    
    def getArg(ind, defValue):
        if ind < len(args):
            return args[ind]
        return defValue
    
    originalExposurePath = getArg(0, DefOriginalExposurePath)
    warpedWcsImageOrExposurePath = getArg(1, DefWcsImageOrExposurePath)
    outputExposurePath = getArg(2, DefOutputExposurePath)
    print "Remapping masked image  ", originalExposurePath
    print "to match wcs and size of", warpedWcsImageOrExposurePath
    print "using", kernelName, "kernel"
    
    warpingControl = afwMath.WarpingControl(kernelName)
    
    originalExposure = afwImage.ExposureF(originalExposurePath)
    
    if warpedWcsImageOrExposurePath.lower().endswith(".fits"):
        # user specified an image, not an exposure
        warpedDI = afwImage.DecoratedImageF(warpedWcsImageOrExposurePath)
        warpedWcs = afwImage.makeWcs(warpedDI.getMetadata())
        warpedMI = afwImage.MaskedImageF(afwGeom.Extent2I(warpedDI.getWidth(), warpedDI.getHeight()))
        warpedExposure = afwImage.ExposureF(warpedMI, warpedWcs)
    else:
        warpedExposure = afwImage.ExposureF(warpedWcsImageOrExposurePath)
    
    if opt.verbosity > 0:
        print "Verbosity =", opt.verbosity
        lsst.pex.logging.Trace_setVerbosity("lsst.afw.math", opt.verbosity)
    
    numGoodPixels = afwMath.warpExposure(warpedExposure, originalExposure, warpingControl)
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
