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
from __future__ import print_function
from builtins import input
import sys
import matplotlib.pyplot as plt

import lsst.pex.logging as pexLog
import lsst.daf.persistence as dafPersist
import lsst.afw.cameraGeom.utils as cameraGeomUtils

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Show the layout of CCDs in a camera.',
                                     epilog=
                        'The corresponding obs-package must be setup (e.g. obs_decam if you want to see DECam)'
                                     )
    parser.add_argument('mapper', help="Name of camera (e.g. decam)", default=None)
    parser.add_argument('--outputFile', type=str, help="File to write plot to", default=None)
    parser.add_argument('--ids', action="store_true", help="Use CCD's IDs, not names")

    args = parser.parse_args()

    #
    # Import the obs package and lookup the mapper
    #
    obsPackageName = "lsst.obs.%s" % args.mapper # guess the package

    try:
        __import__(obsPackageName)
    except:
        print("Unable to import %s -- is it setup?" % (obsPackageName,), file=sys.stderr)
        sys.exit(1)

    obsPackage = sys.modules[obsPackageName] # __import__ returns the top-level module, so look ours up

    mapperName = "%s%sMapper" % (args.mapper[0].title(), args.mapper[1:]) # guess the name too
    try:
        mapper = getattr(obsPackage, mapperName)
    except AttributeError:
        print("Unable to find mapper %s in %s" % (mapperName, obsPackageName), file=sys.stderr)
        sys.exit(1)
    #
    # Control verbosity from butler
    #
    log = pexLog.Log.getDefaultLog()
    log.setThresholdFor("CameraMapper", pexLog.Log.FATAL)
    #
    # And finally find the camera
    #
    camera = mapper().camera

    if not args.outputFile:
        plt.interactive(True)

    cameraGeomUtils.plotFocalPlane(camera, useIds=args.ids,
                                   showFig=not args.outputFile, savePath=args.outputFile)

    if not args.outputFile:
        print("Hit any key to exit", end=' '); input()

    sys.exit(0)
