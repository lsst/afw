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
An example using SpatialCells

Run with:
   python SpatialCellExample.py
or
   python
   >>> import SpatialCellExample; SpatialCellExample.run()
"""

import os
import sys

import lsst.utils
import lsst.afw.detection as afwDetect
import lsst.afw.image.imageLib as afwImage
import lsst.afw.math.mathLib as afwMath
import lsst.afw.geom as afwGeom
import lsst.afw.display.ds9 as ds9

import testSpatialCellLib

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def readImage(filename=None):
    """Read an image and background subtract it"""
    if not filename:
        try:
            afwDataDir = lsst.utils.getPackageDir("afwdata")
        except Exception:
            raise RuntimeError("You must provide a filename or setup afwdata to run these examples")

        filename = os.path.join(afwDataDir, "CFHT", "D4", "cal-53535-i-797722_1")

        bbox = afwGeom.Box2I(afwGeom.Point2I(270, 2530), afwGeom.Extent2I(512, 512))
    else:
        bbox = None

    mi = afwImage.MaskedImageF(filename, 0, None, bbox, afwImage.LOCAL)
    mi.setXY0(afwGeom.Point2I(0, 0))
    #
    # Subtract the background.  We'd use a canned procedure, but that's in meas/utils/sourceDetection.py. We
    # can't fix those pesky cosmic rays either, as that's in a dependent product (meas/algorithms) too
    #
    bctrl = afwMath.BackgroundControl(afwMath.Interpolate.NATURAL_SPLINE)
    bctrl.setNxSample(int(mi.getWidth()/256) + 1)
    bctrl.setNySample(int(mi.getHeight()/256) + 1)
    sctrl = bctrl.getStatisticsControl()
    sctrl.setNumSigmaClip(3.0)
    sctrl.setNumIter(2)

    im = mi.getImage()
    try:
        backobj = afwMath.makeBackground(im, bctrl)
    except Exception, e:
        print >> sys.stderr, e,

        bctrl.setInterpStyle(afwMath.Interpolate.CONSTANT)
        backobj = afwMath.makeBackground(im, bctrl)

    im -= backobj.getImageF()
    #
    # Find sources
    #
    threshold = afwDetect.Threshold(5, afwDetect.Threshold.STDEV)
    npixMin = 5                         # we didn't smooth
    fs = afwDetect.FootprintSet(mi, threshold, "DETECTED", npixMin)
    grow, isotropic = 1, False
    fs = afwDetect.FootprintSet(fs, grow, isotropic)
    fs.setMask(mi.getMask(), "DETECTED")

    return mi, fs

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def SpatialCellSetDemo(filename=None):
    """A demonstration of the use of a SpatialCellSet"""

    im, fs = readImage(filename)

    if display:
        ds9.mtv(im, frame=0, title="Input")
    #
    # Create an (empty) SpatialCellSet
    #
    cellSet = afwMath.SpatialCellSet(afwGeom.Box2I(afwGeom.Point2I(0, 0), im.getDimensions()),
                                     260, 200)

    if display:
        for i in range(len(cellSet.getCellList())):
            cell = cellSet.getCellList()[i]
            ds9.line([(cell.getBBox().getMinX(), cell.getBBox().getMinY()),
                      (cell.getBBox().getMinX(), cell.getBBox().getMaxY()),
                      (cell.getBBox().getMaxX(), cell.getBBox().getMaxY()),
                      (cell.getBBox().getMaxX(), cell.getBBox().getMinY()),
                      (cell.getBBox().getMinX(), cell.getBBox().getMinY()),
                      ], frame=0)
            ds9.dot(cell.getLabel(),
                    (cell.getBBox().getMinX() + cell.getBBox().getMaxX())/2,
                    (cell.getBBox().getMinY() + cell.getBBox().getMaxY())/2)
    #
    # Populate cellSet
    #
    for foot in fs.getFootprints():
        bbox = foot.getBBox()
        xc = (bbox.getMinX() + bbox.getMaxX())/2.0
        yc = (bbox.getMinY() + bbox.getMaxY())/2.0
        tc = testSpatialCellLib.ExampleCandidate(xc, yc, im, bbox)
        cellSet.insertCandidate(tc)
    #
    # OK, the SpatialCellList is populated.  Let's do something with it
    #
    visitor = testSpatialCellLib.ExampleCandidateVisitor()

    cellSet.visitCandidates(visitor)
    print "There are %d candidates" % (visitor.getN())

    ctypes = ["red", "yellow", "cyan", ]
    for i in range(cellSet.getCellList().size()):
        cell = cellSet.getCellList()[i]
        cell.visitCandidates(visitor)

        j = 0
        for cand in cell:
            #
            # Swig doesn't know that we're a SpatialCellImageCandidate;  all it knows is that we have
            # a SpatialCellCandidate so we need an explicit (dynamic) cast
            #
            cand = testSpatialCellLib.cast_ExampleCandidate(cand)

            w, h = cand.getBBox().getDimensions()
            if w*h < 75:
                #print "%d %5.2f %5.2f %d" % (i, cand.getXCenter(), cand.getYCenter(), w*h)
                cand.setStatus(afwMath.SpatialCellCandidate.BAD)

                if display:
                    ds9.dot("o", cand.getXCenter(), cand.getYCenter(), size=4, ctype=ctypes[i%len(ctypes)])
            else:
                if display:
                    ds9.dot("%s:%d" % (cand.getId(), j),
                            cand.getXCenter(), cand.getYCenter(), size=4, ctype=ctypes[i%len(ctypes)])
            j += 1

            im = cand.getMaskedImage()
            if 0 and display:
                ds9.mtv(im, title="Candidate", frame=1)
    #
    # Now count the good and bad candidates
    #
    for i in range(len(cellSet.getCellList())):
        cell = cellSet.getCellList()[i]
        cell.visitCandidates(visitor)

        cell.setIgnoreBad(False)        # include BAD in cell.size()
        print "%s nobj=%d N_good=%d NPix_good=%d" % \
              (cell.getLabel(), cell.size(), visitor.getN(), visitor.getNPix())


    cellSet.setIgnoreBad(True)           # don't visit BAD candidates
    cellSet.visitCandidates(visitor)
    print "There are %d good candidates" % (visitor.getN())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def run():
    """Run the tests"""

    SpatialCellSetDemo()

if __name__ == "__main__":
    run()
