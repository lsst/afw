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

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.daf.base as dafBase
import lsst.pex.logging

def main():
    DefKernel = "lanczos4"
    DefVerbosity = 1

    usage = """usage: %%prog [options] srcExposure refExposure destExposure

Computes destExposure = srcExposure warped to match refExposure's WCS and bounding box,
where exposure arguments are paths to Exposure fits files"""

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

    if len(args) != 3:
        parser.error("You must supply three arguments")

    srcExposurePath = args[0]
    refExposurePath = args[1]
    destExposurePath = args[2]
    print "Remapping exposure      :", srcExposurePath
    print "to match wcs and bbox of:", refExposurePath
    print "using", kernelName, "kernel"

    warpingControl = afwMath.WarpingControl(kernelName)

    srcExposure = afwImage.ExposureF(srcExposurePath)

    destExposure = afwImage.ExposureF(refExposurePath)

    if opt.verbosity > 0:
        print "Verbosity =", opt.verbosity
        lsst.pex.logging.Trace_setVerbosity("lsst.afw.math", opt.verbosity)

    numGoodPixels = afwMath.warpExposure(destExposure, srcExposure, warpingControl)
    print "Warped exposure has %s good pixels" % (numGoodPixels)

    print "Writing warped exposure to %s" % (destExposurePath,)
    destExposure.writeFits(destExposurePath)

if __name__ == "__main__":
    memId0 = dafBase.Citizen_getNextMemId()
    main()
    # check for memory leaks
    if dafBase.Citizen_census(0, memId0) != 0:
        print dafBase.Citizen_census(0, memId0), "Objects leaked:"
        print dafBase.Citizen_census(dafBase.cout, memId0)
