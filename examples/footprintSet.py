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

"""
Examples of using Footprints
"""
from __future__ import print_function

import sys
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.detection as afwDetect
import lsst.afw.display.ds9 as ds9

def showPeaks(im=None, fs=None, frame=0):
    """Show the image and peaks"""
    if frame is None:
        return

    if im:
        ds9.mtv(im, frame=frame)

    if fs:
        with ds9.Buffering():           # turn on buffering of ds9's slow "region" writes
            for foot in fs.getFootprints():
                for p in foot.getPeaks():
                    ds9.dot("+", p.getIx(), p.getIy(), size=0.4, ctype=ds9.RED, frame=frame)

def run(frame=6):
    im = afwImage.MaskedImageF(afwGeom.Extent2I(14, 10))
    #
    # Populate the image with objects that we should detect
    #
    objects = []
    objects.append([(4, 1, 10), (3, 2, 10), (4, 2, 20), (5, 2, 10), (4, 3, 10),])
    objects.append([(9, 7, 30), (10, 7, 29), (12, 7, 28), (10, 8, 27), (11, 8, 26), (10, 4, -5)])
    objects.append([(3, 8, 10), (4, 8, 10),])

    for obj in objects:
        for x, y, I in obj:
            im.getImage().set(x, y, I)

    im.getVariance().set(1)
    im.getVariance().set(10, 4, 0.5**2)
    #
    # Detect the objects at 10 counts or above
    #
    level = 10
    fs = afwDetect.FootprintSet(im, afwDetect.Threshold(level), "DETECTED")

    showPeaks(im, fs, frame=frame)
    #
    # Detect the objects at -10 counts or below.  N.b. the peak's at -5, so it isn't detected
    #
    polarity = False                     # look for objects below background
    threshold = afwDetect.Threshold(level, afwDetect.Threshold.VALUE, polarity)
    fs2 = afwDetect.FootprintSet(im, threshold, "DETECTED_NEGATIVE")
    print("Detected %d objects below background" % len(fs2.getFootprints()))
    #
    # Search in S/N (n.b. the peak's -10sigma)
    #
    threshold = afwDetect.Threshold(level, afwDetect.Threshold.PIXEL_STDEV, polarity)
    fs2 = afwDetect.FootprintSet(im, threshold)
    #
    # Here's another way to set a mask plane (we chose not to do so in the FootprintSet call)
    #
    msk = im.getMask()
    afwDetect.setMaskFromFootprintList(msk, fs2.getFootprints(), msk.getPlaneBitMask("DETECTED_NEGATIVE"))

    if frame is not None:
        ds9.mtv(msk, isMask=True, frame=frame)
    #
    # Merge the positive and negative detections, growing both sets by 1 pixel
    #
    fs.merge(fs2, 1, 1)
    #
    # Set EDGE so we can see the grown Footprints
    #
    afwDetect.setMaskFromFootprintList(msk, fs.getFootprints(), msk.getPlaneBitMask("EDGE"))

    if frame is not None:
        ds9.mtv(msk, isMask=True, frame=frame)

    showPeaks(fs=fs, frame=frame)

if __name__ == "__main__":
    run(frame=None)
