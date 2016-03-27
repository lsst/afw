#!/usr/bin/env python
#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#
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
        print >> sys.stderr, "Unable to import %s -- is it setup?" % (obsPackageName,)
        sys.exit(1)

    obsPackage = sys.modules[obsPackageName] # __import__ returns the top-level module, so look ours up

    mapperName = "%s%sMapper" % (args.mapper[0].title(), args.mapper[1:]) # guess the name too
    try:
        mapper = getattr(obsPackage, mapperName)
    except AttributeError:
        print >> sys.stderr, "Unable to find mapper %s in %s" % (mapperName, obsPackageName)
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
        print "Hit any key to exit",; raw_input()

    sys.exit(0)
