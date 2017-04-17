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

from __future__ import absolute_import, division, print_function
from builtins import input
from builtins import zip
import math
import numpy
import matplotlib.pyplot as plt

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.geom as afwGeom

def main(camera, sample=20, showDistortion=True):
    if True:
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))

        title = camera.getId().getName()
        if showDistortion:
            title += ' (Distorted)'

        ax.set_title(title)
    else:
        fig = None

    if showDistortion:
        dist = camera.getDistortion()

    for raft in camera:
        raft = raft
        for ccd in raft:
            if False and ccd.getId().getSerial() not in (0, 3):
                continue

            ccd = ccd
            ccd.setTrimmed(True)

            width, height = ccd.getAllPixels(True).getDimensions()

            corners = ((0.0,0.0), (0.0, height), (width, height), (width, 0.0), (0.0, 0.0))
            for (x0, y0), (x1, y1) in zip(corners[0:4],corners[1:5]):
                if x0 == x1 and y0 != y1:
                    yList = numpy.linspace(y0, y1, num=sample)
                    xList = [x0] * len(yList)
                elif y0 == y1 and x0 != x1:
                    xList = numpy.linspace(x0, x1, num=sample)
                    yList = [y0] * len(xList)
                else:
                    raise RuntimeError("Should never get here")

                xOriginal = []; yOriginal = []
                xDistort = []; yDistort = []
                for x, y in zip(xList, yList):
                    position = ccd.getPositionFromPixel(afwGeom.Point2D(x,y)) # focal plane position

                    xOriginal.append(position.getMm().getX())
                    yOriginal.append(position.getMm().getY())

                    if not showDistortion:
                        continue

                    # Calculate offset (in CCD pixels) due to distortion
                    distortion = dist.distort(afwGeom.Point2D(x, y), ccd) - afwGeom.Extent2D(x, y)

                    # Calculate the distorted position
                    distorted = position + cameraGeom.FpPoint(distortion)*ccd.getPixelSize()

                    xDistort.append(distorted.getMm().getX())
                    yDistort.append(distorted.getMm().getY())

                if fig:
                    ax.plot(xOriginal, yOriginal, 'k-')
                    if showDistortion:
                        ax.plot(xDistort, yDistort, 'r-')

            if fig:
                x,y = ccd.getPositionFromPixel(afwGeom.Point2D(width/2, height/2)).getMm()
                ax.text(x, y, ccd.getId().getSerial(), ha='center')

    if fig:
        plt.show()

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("camera", help="Name of camera to show")
    parser.add_argument("--showDistortion", action="store_true", help="Show distortion?")
    parser.add_argument("-v", "--verbose", action="store_true", help="Be chattier")
    args = parser.parse_args()

    if args.camera.lower() == "hsc":
        from lsst.obs.hscSim.hscSimMapper import HscSimMapper as Mapper
    elif args.camera.lower() == "suprimecam":
        from lsst.obs.suprimecam import SuprimecamMapper as Mapper
    elif args.camera.lower() == "lsstsim":
        from lsst.obs.lsstSim import LsstSimMapper as Mapper
    else:
        print("Unknown camera %s" % args.camera, file=sys.stderr)
        sys.exit(1)

    camera = Mapper().camera

    main(camera, showDistortion=args.showDistortion, sample=2)
    print("Hit any key to exit", end=' '); input()
