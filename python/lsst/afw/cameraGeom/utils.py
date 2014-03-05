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
Support for cameraGeom
"""
from __future__ import division
import math
import os
import sys
import unittest
import numpy as numpy
import itertools
try:
    import pyfits
except ImportError:
    pyfits = None

import lsst.daf.persistence as dafPersist
import lsst.pex.policy as pexPolicy
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.cameraGeom as cameraGeom

import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

from lsst.afw.cameraGeom import CameraSys, PUPIL, PIXELS, FOCAL_PLANE
#TODO protect this or add it to setups.
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
try:
    type(display)
except NameError:
    display = False
    force = False


def plotFocalPlane(camera, pupilSizeDeg_x, pupilSizeDeg_y, dx=0.1, dy=0.1, figsize=(10., 10.), showFig=True, savePath=None):
    """
    Make a plot of the focal plane along with a set points that sample the Pupil
    @param camera -- a camera object
    """
    pupil_gridx, pupil_gridy = numpy.meshgrid(numpy.arange(0., pupilSizeDeg_x+dx, dx) - pupilSizeDeg_x/2., 
                                              numpy.arange(0., pupilSizeDeg_y+dy, dy) -  pupilSizeDeg_y/2.)
    xs = []
    ys = []
    pcolors = []
    for pos in zip(pupil_gridx.flatten(), pupil_gridy.flatten()):
        posRad = afwGeom.Point2D(math.radians(pos[0]), math.radians(pos[1]))
        cp = camera.makeCameraPoint(posRad, PUPIL)
        ncp = camera.transform(cp, FOCAL_PLANE)
        xs.append(ncp.getPoint().getX())
        ys.append(ncp.getPoint().getY())
        dets = camera.findDetectors(cp)
        if len(dets) > 0:
            pcolors.append('w')
        else:
            pcolors.append('k')


    colorMap = {0:'b', 1:'y', 2:'g', 3:'r'}

    patches = []
    colors = []
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    xvals = []
    yvals = []
    for det in camera:
        corners = [(c.getX(), c.getY()) for c in det.getCorners(FOCAL_PLANE)]
        for corner in corners:
            xvals.append(corner[0])
            yvals.append(corner[1])
        colors.append(colorMap[det.getType()])
        patches.append(Polygon(corners, True))
        center = det.getOrientation().getFpPosition()
        ax.text(center.getX(), center.getY(), det.getName(), horizontalalignment='center', size=8)

    p = PatchCollection(patches, alpha=0.6, facecolor=colors)
    ax.add_collection(p)
    ax.scatter(xs, ys, s=10, alpha=.7, linewidths=0., c=pcolors)
    ax.set_xlim(min(xvals) - abs(0.1*min(xvals)), max(xvals) + abs(0.1*max(xvals)))
    ax.set_ylim(min(yvals) - abs(0.1*min(yvals)), max(yvals) + abs(0.1*max(yvals)))
    ax.set_xlabel('Focal Plane X (mm)')
    ax.set_ylabel('Focal Plane Y (mm)')
    if savePath is not None:
        plt.savefig(savePath)
    if showFig:
        plt.show()

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def makeImageFromAmp(amp, imValue=None, imageFactory=afwImage.ImageU, markSize=10, markValue=0):
    if not amp.getHasRawInfo():
        raise RuntimeError("Can't create a raw amp image without raw amp data")
    bbox = amp.getRawBBox()
    dbbox = amp.getRawDataBBox()
    img = imageFactory(bbox)
    if imValue is None:
        img.set((amp.getGain()*1000)//10)
    else:
        img.set(imValue)
    #TODO set the first pixel read to a different value
    markbbox = afwGeom.Box2I()
    if amp.getReadoutCorner() == 0:
        markbbox.include(dbbox.getMin())
        markbbox.include(dbbox.getMin()+afwGeom.Extent2I(markSize, markSize))
    elif amp.getReadoutCorner() == 1:
        cornerPoint = afwGeom.Point2I(dbbox.getMaxX(), dbbox.getMinY())
        markbbox.include(cornerPoint)
        markbbox.include(cornerPoint + afwGeom.Extent2I(-markSize, markSize))
    elif amp.getReadoutCorner() == 2:
        cornerPoint = afwGeom.Point2I(dbbox.getMax())
        markbbox.include(cornerPoint)
        markbbox.include(cornerPoint + afwGeom.Extent2I(-markSize, -markSize))
    elif amp.getReadoutCorner() == 3:
        cornerPoint = afwGeom.Point2I(dbbox.getMinX(), dbbox.getMaxY())
        markbbox.include(cornerPoint)
        markbbox.include(cornerPoint + afwGeom.Extent2I(markSize, -markSize))
    else:
        raise RuntimeError("Could not set readout corner")
    mimg = imageFactory(img, markbbox, False)
    mimg.set(markValue)
    return img

def calcRawCcdBBox(ccd):
    bbox = afwGeom.Box2I()
    for amp in ccd:
        if not amp.getHasRawInfo():
            raise RuntimeError("Cannot build a raw CCD bounding box without raw amp information")
        tbbox = amp.getRawBBox()
        tbbox.shift(amp.getRawXYOffset())
        bbox.include(tbbox)
    return bbox

def makeImageFromCcd(ccd, isTrimmed=True, showAmpGain=True, imageFactory=afwImage.ImageU, rcMarkSize=10, bin=1):
    """Make an Image of a Ccd
    """
    ampImages = []
    index = 0
    if isTrimmed:
         bbox = ccd.getBBox()
    else:
         bbox = calcRawCcdBBox(ccd)
    for amp in ccd:
        if amp.getHasRawInfo():
            if showAmpGain:
                ampImages.append(makeImageFromAmp(amp, imageFactory=imageFactory, markSize=rcMarkSize))
            else:
                ampImages.append(makeImageFromAmp(amp, imValue=(index+1)*1000, imageFactory=imageFactory, markSize=rcMarkSize))
            index += 1

    if len(ampImages) > 0:
        ccdImage = imageFactory(bbox)
        for ampImage, amp in itertools.izip(ampImages, ccd):
            if isTrimmed:
                cameraGeom.assembleAmplifierImage(ccdImage, ampImage, amp)
            else:
                cameraGeom.assembleAmplifierRawImage(ccdImage, ampImage, amp)
    else:
        if not isTrimmed:
            raise RuntimeError("Cannot create untrimmed CCD without amps with raw information")
        ccdImage = imageFactory(ccd.getBBox())
    ccdImage = afwMath.binImage(ccdImage, bin)
    return ccdImage

def overlayCcdBoxes(ccd, untrimmedCcdBbox, nQuarter, isTrimmed, ccdOrigin, frame, bin):
    with ds9.Buffering():
        ccdDim = untrimmedCcdBbox.getDimensions()
        ccdBbox = cameraGeom.rotateBBoxBy90(untrimmedCcdBbox, nQuarter, ccdDim)
        for amp in ccd:
            if isTrimmed:
                ampbbox = amp.getBBox()
            else:
                ampbbox = amp.getRawBBox()
                ampbbox.shift(amp.getRawXYOffset())
            if nQuarter != 0:
                ampbbox = cameraGeom.rotateBBoxBy90(ampbbox, nQuarter, ccdDim)

            displayUtils.drawBBox(ampbbox, origin=ccdOrigin, borderWidth=0.49,
                                  frame=frame, bin=bin)

            if not isTrimmed and amp.getHasRawInfo():
                for bbox, ctype in ((amp.getRawHorizontalOverscanBBox(), ds9.RED), (amp.getRawDataBBox(), ds9.BLUE),
                                    (amp.getRawVerticalOverscanBBox(), ds9.MAGENTA), (amp.getRawPrescanBBox(), ds9.YELLOW)):
                    if amp.getRawFlipX():
                        bbox.flipLR(amp.getRawBBox().getDimensions().getX())
                    if amp.getRawFlipY():
                        bbox.flipTB(amp.getRawBBox().getDimensions().getY())
                    bbox.shift(amp.getRawXYOffset())
                    if nQuarter != 0:
                        bbox = cameraGeom.rotateBBoxBy90(bbox, nQuarter, ccdDim)
                    displayUtils.drawBBox(bbox, origin=ccdOrigin, borderWidth=0.49, ctype=ctype, frame=frame, bin=bin)
            # Label each Amp
            xc, yc = (ampbbox.getMin()[0] + ampbbox.getMax()[0])//2, (ampbbox.getMin()[1] +
                    ampbbox.getMax()[1])//2
            cen = afwGeom.Point2I(xc, yc)
            #
            # Rotate the amp labels too
            #
            if nQuarter == 0:
                c, s = 1, 0
            elif nQuarter == 1:
                c, s = 0, -1
            elif nQuarter == 2:
                c, s = -1, 0
            elif nQuarter == 3:
                c, s = 0, 1
            c, s = 1, 0
            ccdHeight = ccdBbox.getHeight()
            ccdWidth = ccdBbox.getWidth()
            xc -= 0.5*ccdHeight
            yc -= 0.5*ccdWidth

            xc, yc = 0.5*ccdHeight + c*xc + s*yc, 0.5*ccdWidth + -s*xc + c*yc

            if ccdOrigin:
                xc += ccdOrigin[0]
                yc += ccdOrigin[1]
            ds9.dot(str(amp.getName()), xc/bin, yc/bin, frame=frame)

        displayUtils.drawBBox(ccdBbox, origin=ccdOrigin,
                              borderWidth=0.49, ctype=ds9.MAGENTA, frame=frame, bin=bin)

def showAmp(amp, ampImage=None, isTrimmed=False, frame=None, overlay=True, imageFactory=afwImage.ImageU, markSize=10, markValue=0):
    if ampImage is None:
        ampImage = makeImageFromAmp(amp, imageFactory=imageFactory, markSize=markSize, markValue=markValue)
    else:
        if isTrimmed and not ampImage.getBBox() == amp.getBBox():
            raise ValueError("Image is not same size as amp bounding box: %s -- %s"%(ampImage.getBBox(), amp.getBBox()))
        if not isTrimmed and not ampImage.getBBox() == amp.getRawBBox():
            raise ValueError("Image is not same size as amp bounding box: %s -- %s"%(ampImage.getBBox(), amp.getRawBBox()))
    title = amp.getName()
    if isTrimmed:
        ampImage = ampImage.Factory(ampImage, amp.getRawDataBBox(), False)
        ds9.mtv(ampImage, frame=frame, title=title)
    else:
        ds9.mtv(ampImage, frame=frame, title=title)
    if overlay:
        with ds9.Buffering():
            if not isTrimmed:
                bboxes = [(amp.getRawBBox(), 0.49, ds9.GREEN),]
                xy0 = bboxes[0][0].getMin()
                bboxes.append((amp.getRawHorizontalOverscanBBox(), 0.49, ds9.RED)) 
                bboxes.append((amp.getRawDataBBox(), 0.49, ds9.BLUE))
                bboxes.append((amp.getRawPrescanBBox(), 0.49, ds9.YELLOW))
                bboxes.append((amp.getRawVerticalOverscanBBox(), 0.49, ds9.MAGENTA))
            else:
                bboxes = [(amp.getBBox(), 0.49, None),]
                xy0 = bboxes[0][0].getMin()

            for bbox, borderWidth, ctype in bboxes:
                if bbox.isEmpty():
                    continue
                bbox = afwGeom.Box2I(bbox)
                bbox.shift(-afwGeom.ExtentI(xy0))
                displayUtils.drawBBox(bbox, borderWidth=borderWidth, ctype=ctype, frame=frame)

def showCcd(ccd, ccdImage=None, isTrimmed=True, showAmpGain=True, frame=None, overlay=True, bin=1, inCameraCoords=False):
    """Show a CCD on ds9.  If ccdImage is None, an image will be created based on the properties
of the detectors"""
    ccdOrigin = afwGeom.Point2I(0,0)
    nQuarter = 0
    if ccdImage is None:
        ccdImage = makeImageFromCcd(ccd, isTrimmed=isTrimmed, showAmpGain=showAmpGain, bin=bin)
    else:
        rawBbox = calcRawCcdBBox(ccd)
        if isTrimmed and not ccdImage.getBBox() == ccd.getBBox():
            raise ValueError("Image is not same size as amp bounding box: %s -- %s"%(ampImage.getBBox(), ccd.getBBox()))
        if not isTrimmed and not ccdImage.getBBox() == rawBbox:
            raise ValueError("Image is not same size as amp bounding box: %s -- %s"%(ampImage.getBBox(), rawBbox))
    ccdBbox = ccdImage.getBBox()
    if inCameraCoords:
        #TODO add getNQuarter to detector
        yaw = ccd.getOrientation().getYaw()
        while (yaw.asDegrees() < 0.):
            yaw += afwGeom.Angle(afwGeom.TWOPI)
        nQuarter = (yaw+afwGeom.Angle(afwGeom.PI/4.)).asDegrees()//90
        nQuarter = int(nQuarter)
        ccdImage = afwMath.rotateImageBy90(ccdImage, nQuarter)
    title = ccd.getName()
    if isTrimmed:
        title += "(trimmed)"
    ds9.mtv(ccdImage, frame=frame, title=title)

    if overlay:
        overlayCcdBoxes(ccd, ccdBbox, nQuarter, isTrimmed, ccdOrigin, frame, bin)


def makeImageFromCamera(camera, detectorList=None, background=numpy.nan, buffer=10, imageSource=None, imageFactory=afwImage.ImageU, bin=1):
    """Make an Image of a Camera"""
    if detectorList is None:
        detectorList = camera._nameDetectorDict.keys()
    
    camBbox = camera.getFpBBox()
    pixelSize_o = camera[0].getPixelSize()
    pixMin = afwGeom.Point2I(int(camBbox.getMinX()//pixelSize_o.getX()), int(camBbox.getMinY()//pixelSize_o.getY()))
    pixMax = afwGeom.Point2I(int(camBbox.getMaxX()//pixelSize_o.getX()), int(camBbox.getMaxY()//pixelSize_o.getY()))
    camBbox = afwGeom.Box2I(pixMin, pixMax)
    camBbox.grow(buffer)
    origin = camBbox.getMin()
    # This segfaults for large images.  It seems better to throw instead of segfaulting, but maybe that's not easy.
    camIm = imageFactory(int(camBbox.getDimensions().getX()/bin), int(camBbox.getDimensions().getY()/bin))
    
    for det in (camera[name] for name in detectorList):
        if not pixelSize_o == det.getPixelSize():
            raise RuntimeError("Cameras with detectors with different pixel scales are not currently supported")
        if imageSource is None:
            im = makeImageFromCcd(det, isTrimmed=True, showAmpGain=False, imageFactory=imageFactory, bin=bin)
        else:
            raise NotImplementedError("Do something reasonable if an image is sent")

        #TODO add getNQuarter to detector -- could break this out into util.
        yaw = det.getOrientation().getYaw()
        while (yaw.asDegrees() < 0.):
            yaw += afwGeom.Angle(afwGeom.TWOPI)
        nQuarter = (yaw+afwGeom.Angle(afwGeom.PI/4.)).asDegrees()//90
        nQuarter = int(nQuarter)
        im = afwMath.rotateImageBy90(im, nQuarter)
        dbbox = afwGeom.Box2D()
        for corner in det.getCorners(FOCAL_PLANE):
            dbbox.include(corner)
        llc = dbbox.getMin()
        bbox = im.getBBox()
        bbox.shift(afwGeom.Extent2I(int(llc.getX()//pixelSize_o.getX()/bin), int(llc.getY()//pixelSize_o.getY()/bin)))
        bbox.shift(afwGeom.Extent2I(-int(origin.getX()//bin), -int(origin.getY())//bin))
        imView = camIm.Factory(camIm, bbox, afwImage.LOCAL)
        imView <<= im

    return camIm

def showCamera(camera, imageSource=None, imageFactory=afwImage.ImageU, detectorList=None,
                bin=10, buffer=10, frame=None, overlay=True, title="", ctype=ds9.GREEN, 
                referenceDetectorName=None, **kwargs):
    """Show a Camera on ds9 (with the specified frame); if overlay show the IDs and detector boundaries

If imageSource is provided its getImage method will be called to return a CCD image (e.g. a
cameraGeom.GetCcdImage object); if it is "", an image will be created based on the properties
of the detectors"""
    
    cameraImage = makeImageFromCamera(camera, detectorList=detectorList, buffer=buffer,
                                      imageSource=imageSource, imageFactory=imageFactory, bin=bin)
    wcs = makeFocalPlaneWcs(camera, bin, referenceDetectorName)
    #TODO makeFocalPlaneWcs is returning None and I don't know why.
    ds9.mtv(cameraImage, title=title, frame=frame, wcs=wcs)
   
    return cameraImage

def makeFocalPlaneWcs(camera, bin=1, referenceDetectorName=None):
    """Make a WCS for the focal plane geometry (i.e. returning positions in "mm")"""
    import lsst.daf.base as dafBase

    if referenceDetectorName is not None:
        ccd = camera[referenceDetectorName]
    else:
        cp = camera.makeCameraPoint(camera.getFpBBox().getCenter(), FOCAL_PLANE)
        ccds = camera.findDetectors(cp)
        if len(ccds) == 1:
            ccd = ccds[0]
        else:
            raise RuntimeError("Could not find detector to make WCS.  Specify a detector to use")
    md = dafBase.PropertySet()
    pix = afwGeom.PointD(0,0)
    cp = ccd.makeCameraPoint(pix, PIXELS)
    fpPos = ccd.transform(cp, FOCAL_PLANE)
    fpPos = fpPos.getPoint()
    for i in range(2):
        md.set("CRPOS%d" % i, pix[i])
        md.set("CRVAL%d" % i, fpPos[i])
    md.set("CDELT1", ccd.getPixelSize()[0]*bin)
    md.set("CDELT2", ccd.getPixelSize()[1]*bin)

    return afwImage.makeWcs(md)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#TODO fix this, but maybe not until S14???
def showMosaic(fileName, geomPolicy=None, camera=None,
               display=True, what=cameraGeom.Camera, id=None, overlay=False, describe=False, doTrim=False,
               imageFactory=afwImage.ImageU, bin=1, frame=None):
    """Return a mosaic for a given snapshot of the sky; if display is true, also show it on ds9

The camera geometry is defined by cameraGeomPolicyFile;  raft IDs etc. are drawn on ds9 if overlay is True;
The camera (or raft) is described if describe is True

You may set what to a type (e.g. cameraGeom.Ccd) to display that type; if provided id will be obeyed

If relevant (for e.g. a Ccd) doTrim is applied to the Detector.
    """

    if isinstance(fileName, GetCcdImage):
        imageSource = fileName
    elif isinstance(fileName, str):
        imageSource = GetCcdImage(fileName) # object that understands the CCD <--> HDU mapping
    else:
        imageSource = None

    if imageSource:
        imageSource.setTrimmed(doTrim)
    
    if not camera:
        camera = makeCamera(geomPolicy)

    if what == cameraGeom.Amp:
        if id is None:
            ccd = makeCcd(geomPolicy)
        else:
            ccd = findCcd(camera, id[0])
        amp = [a for a in ccd if a.getId() == id[1]][0]

        if not amp:
            raise RuntimeError, "Failed to find Amp %s" % id

        ccd.setTrimmed(doTrim)

        outImage = makeImageFromCcd(ccd, imageSource, amp=amp, imageFactory=imageFactory, bin=bin)
        if display:
            showCcd(ccd, outImage, amp=amp, overlay=overlay, frame=frame, bin=bin)
    elif what == cameraGeom.Ccd:
        if id is None:
            ccd = makeCcd(geomPolicy)
        else:
            ccd = findCcd(camera, id)

        if not ccd:
            raise RuntimeError, "Failed to find Ccd %s" % id

        ccd.setTrimmed(doTrim)

        outImage = makeImageFromCcd(ccd, imageSource, imageFactory=imageFactory, bin=bin)
        if display:
            showCcd(ccd, outImage, overlay=overlay, frame=frame, bin=bin)
    elif what == cameraGeom.Raft:
        if id:
            raft = findRaft(camera, id)
        else:
            raft = makeRaft(geomPolicy)
        if not raft:
            raise RuntimeError, "Failed to find Raft %s" % id

        outImage = makeImageFromRaft(raft, imageSource, imageFactory=imageFactory, bin=bin)
        if display:
            showRaft(raft, outImage, overlay=overlay, frame=frame, bin=bin)

        if describe:
            print describeRaft(raft)
    elif what == cameraGeom.Camera:
        outImage = makeImageFromCamera(camera, imageSource, imageFactory=imageFactory, bin=bin)
        if display:
            showCamera(camera, outImage, overlay=overlay, frame=frame, bin=bin)

        if describe:
            print describeCamera(camera)
    else:
        raise RuntimeError, ("I don't know how to display %s" % what)

    return outImage

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def findAmp(ccd, pixelPosition):
    """Find the Amp with the specified Id within the composite"""

    for amp in ccd:
        if amp.getBBox().contains(pixelPosition):
            return amp

    return None
