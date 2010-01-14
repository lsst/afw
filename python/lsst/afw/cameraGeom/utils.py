#!/usr/bin/env python
"""
Tests for SpatialCell

Run with:
   python SpatialCell.py
or
   python
   >>> import SpatialCell; SpatialCell.run()
"""

import math
import os
import sys
import unittest

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.cameraGeom as cameraGeom

import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class GetCcdImage(object):
    """A class to return an Image of a given Ccd"""

    def __init__(self, imageFile):
        self.imageFile = imageFile

    def getImage(self, id, bbox=None, imageFactory=afwImage.ImageU):
        """Return the image of the chip with cameraGeom.Id == id; if provided only read the given BBox"""

        md = None
        if not bbox:
            bbox = afwImage.BBox()

        return imageFactory(self.imageFile, id.getSerial(), md, bbox)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def makeCcd(geomPolicy, ccdId=None, ccdInfo=None):
    """Build a Ccd from a set of amplifiers given a suitable pex::Policy

If ccdInfo is provided it's set to various facts about the CCDs which are used in unit tests.  Note
in particular that it has an entry ampSerial which is a single-element list, the amplifier serial counter
    """
    ccdPol = geomPolicy.get("Ccd")

    pixelSize = ccdPol.get("pixelSize")

    nCol = ccdPol.get("nCol")
    nRow = ccdPol.get("nRow")
    if not ccdId:
        try:
            ccdId = cameraGeom.Id(ccdPol.get("serial"), ccdPol.get("name"))
        except Exception, e:
            ccdId = cameraGeom.Id(0, "unknown")

    ccd = cameraGeom.Ccd(ccdId, pixelSize)

    if nCol*nRow != len(ccdPol.getArray("Amp")):
        raise RuntimeError, ("Expected location of %d amplifiers, got %d" % \
                             (nCol*nRow, len(ccdPol.getArray("Amp"))))

    if ccdInfo is None:
        ampSerial = [0]
    else:
        ampSerial = ccdInfo.get("ampSerial", [0])
        ampSerial0 = ampSerial[0]           # used in testing
        
    readoutCorners = dict(LLC = cameraGeom.Amp.LLC,
                          LRC = cameraGeom.Amp.LRC,
                          ULC = cameraGeom.Amp.ULC,
                          URC = cameraGeom.Amp.URC)
    for ampPol in ccdPol.getArray("Amp"):
        Col, Row = ampPol.getArray("index")
        c =  ampPol.get("readoutCorner")

        if Col not in range(nCol) or Row not in range(nRow):
            raise RuntimeError, ("Amp location %d, %d is not in 0..%d, 0..%d" % (Col, Row, nCol, nRow))

        gain = ampPol.get("Electronic.gain")
        readNoise = ampPol.get("Electronic.readNoise")
        saturationLevel = ampPol.get("Electronic.saturationLevel")
        #
        # Now lookup properties common to all the CCD's amps
        #
        ampPol = geomPolicy.get("Amp")
        width = ampPol.get("width")
        height = ampPol.get("height")

        extended = ampPol.get("extended")
        preRows = ampPol.get("preRows")
        overclockH = ampPol.get("overclockH")
        overclockV = ampPol.get("overclockV")

        eWidth = extended + width + overclockH
        eHeight = preRows + height + overclockV

        allPixels = afwImage.BBox(afwImage.PointI(0, 0), eWidth, eHeight)

        try:
            c = readoutCorners[c]
        except IndexError:
            raise RuntimeError, ("Unknown readoutCorner %s" % c)
        
        if c in (cameraGeom.Amp.LLC, cameraGeom.Amp.ULC):
            biasSec = afwImage.BBox(afwImage.PointI(extended + width, preRows), overclockH, height)
            dataSec = afwImage.BBox(afwImage.PointI(extended, preRows), width, height)
        elif c in (cameraGeom.Amp.LRC, cameraGeom.Amp.URC):
            biasSec = afwImage.BBox(afwImage.PointI(0, preRows), overclockH, height)
            dataSec = afwImage.BBox(afwImage.PointI(overclockH, preRows), width, height)

        eParams = cameraGeom.ElectronicParams(gain, readNoise, saturationLevel)
        amp = cameraGeom.Amp(cameraGeom.Id(ampSerial[0], "ID%d" % ampSerial[0]),
                             allPixels, biasSec, dataSec, c, eParams)
        ampSerial[0] += 1

        ccd.addAmp(Col, Row, amp)
    #
    # Information for the test code
    #
    if ccdInfo is not None:
        ccdInfo.clear()
        ccdInfo["ampSerial"] = ampSerial
        ccdInfo["name"] = ccd.getId().getName()
        ccdInfo["ampWidth"], ccdInfo["ampHeight"] = width, height
        ccdInfo["width"], ccdInfo["height"] = nCol*eWidth, nRow*eHeight
        ccdInfo["trimmedWidth"], ccdInfo["trimmedHeight"] = nCol*width, nRow*height
        ccdInfo["pixelSize"] = pixelSize
        ccdInfo["ampIdMin"] = ampSerial0
        ccdInfo["ampIdMax"] = ampSerial[0] - 1

    return ccd

def makeRaft(geomPolicy, raftId=None, raftInfo=None):
    """Build a Raft from a set of CCDs given a suitable pex::Policy
    
If raftInfo is provided it's set to various facts about the Rafts which are used in unit tests.  Note in
particular that it has an entry ampSerial which is a single-element list, the amplifier serial counter
"""

    if raftInfo is None:
        ccdInfo = None
    else:
        ccdInfo = {"ampSerial" : raftInfo.get("ampSerial", [0])}

    raftPol = geomPolicy.get("Raft")
    nCol = raftPol.get("nCol")
    nRow = raftPol.get("nRow")
    if not raftId:
        try:
            raftId = cameraGeom.Id(raftPol.get("serial"), raftPol.get("name"))
        except Exception, e:
            raftId = cameraGeom.Id(0, "unknown")

    raft = cameraGeom.Raft(raftId, nCol, nRow)

    if nCol*nRow != len(raftPol.getArray("Ccd")):
        raise RuntimeError, ("Expected location of %d amplifiers, got %d" % \
                             (nCol*nRow, len(raftPol.getArray("Ccd"))))

    for ccdPol in raftPol.getArray("Ccd"):
        Col, Row = ccdPol.getArray("index")
        xc, yc = ccdPol.getArray("offset")

        nQuarter = ccdPol.get("nQuarter")
        pitch, roll, yaw = [float(math.radians(a)) for a in ccdPol.getArray("orientation")]

        if Col not in range(nCol) or Row not in range(nRow):
            raise RuntimeError, ("Amp location %d, %d is not in 0..%d, 0..%d" % (Col, Row, nCol, nRow))

        ccdId = cameraGeom.Id(ccdPol.get("serial"), ccdPol.get("name"))
        ccd = makeCcd(geomPolicy, ccdId, ccdInfo=ccdInfo)

        raft.addDetector(afwGeom.Point2I.makeXY(Col, Row),
                         afwGeom.Point2D.makeXY(xc, yc),
                         cameraGeom.Orientation(nQuarter, pitch, roll, yaw), ccd)

        if raftInfo is not None:
            # Guess the gutter between detectors
            if (Col, Row) == (0, 0):
                xGutter, yGutter = xc, yc
            elif (Col, Row) == (nCol - 1, nRow - 1):
                if nCol == 1:
                    xGutter = 0.0
                else:
                    xGutter = (xc - xGutter)/float(nCol - 1) - ccd.getSize()[0]

                if nRow == 1:
                    yGutter = 0.0
                else:
                    yGutter = (yc - yGutter)/float(nRow - 1) - ccd.getSize()[1]

    if raftInfo is not None:
        raftInfo.clear()
        raftInfo["ampSerial"] = ccdInfo["ampSerial"]
        raftInfo["name"] = raft.getId().getName()
        raftInfo["pixelSize"] = ccd.getPixelSize()
        raftInfo["width"] =  nCol*ccd.getAllPixels(True).getWidth()
        raftInfo["height"] = nRow*ccd.getAllPixels(True).getHeight()
        raftInfo["widthMm"] =  nCol*ccd.getSize()[0] + (nCol - 1)*xGutter
        raftInfo["heightMm"] = nRow*ccd.getSize()[1] + (nRow - 1)*yGutter

    return raft

def makeCamera(geomPolicy, cameraId=None, cameraInfo=None):
    """Build a Camera from a set of Rafts given a suitable pex::Policy
    
If cameraInfo is provided it's set to various facts about the Camera which are used in unit tests.  Note in
particular that it has an entry ampSerial which is a single-element list, the amplifier serial counter
"""
    if cameraInfo is None:
        raftInfo = None
    else:
        raftInfo = {"ampSerial" : cameraInfo.get("ampSerial", [0])}

    cameraPol = geomPolicy.get("Camera")
    nCol = cameraPol.get("nCol")
    nRow = cameraPol.get("nRow")

    if not cameraId:
        cameraId = cameraGeom.Id(cameraPol.get("serial"), cameraPol.get("name"))
    camera = cameraGeom.Camera(cameraId, nCol, nRow)

    for raftPol in cameraPol.getArray("Raft"):
        Col, Row = raftPol.getArray("index")
        xc, yc = raftPol.getArray("offset")
        raftId = cameraGeom.Id(raftPol.get("serial"), raftPol.get("name"))
        raft = makeRaft(geomPolicy, raftId, raftInfo)
        camera.addDetector(afwGeom.Point2I.makeXY(Col, Row),
                           afwGeom.Point2D.makeXY(xc, yc), cameraGeom.Orientation(), raft)

        if cameraInfo is not None:
            # Guess the gutter between detectors
            if (Col, Row) == (0, 0):
                xGutter, yGutter = xc, yc
            elif (Col, Row) == (nCol - 1, nRow - 1):
                if nCol == 1:
                    xGutter = 0.0
                else:
                    xGutter = (xc - xGutter)/float(nCol - 1) - raft.getSize()[0]
                    
                if nRow == 1:
                    yGutter = 0.0
                else:
                    yGutter = (yc - yGutter)/float(nRow - 1) - raft.getSize()[1]

    if cameraInfo is not None:
        cameraInfo.clear()
        cameraInfo["ampSerial"] = raftInfo["ampSerial"]
        cameraInfo["name"] = camera.getId().getName()
        cameraInfo["width"] =  nCol*raft.getAllPixels().getWidth()
        cameraInfo["height"] = nRow*raft.getAllPixels().getHeight()
        cameraInfo["pixelSize"] = raft.getPixelSize()
        cameraInfo["widthMm"] =  nCol*raft.getSize()[0] + (nCol - 1)*xGutter
        cameraInfo["heightMm"] = nRow*raft.getSize()[1] + (nRow - 1)*yGutter

    return camera

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def makeImageFromCcd(ccd, filename=None, isTrimmed=None):
    """Make an Image of a Ccd"""

    if isTrimmed is None:
        isTrimmed = ccd.isTrimmed()

    ccdImage = afwImage.ImageU(ccd.getAllPixels(isTrimmed).getDimensions())
    for a in ccd:
        im = ccdImage.Factory(ccdImage, a.getAllPixels(isTrimmed))
        if filename:
            im <<= filename.getImage(ccd.getId(), a.getDataSec(False))
            continue

        im += int(a.getElectronicParams().getReadNoise())
        im = ccdImage.Factory(ccdImage, a.getDataSec(isTrimmed))
        im += int(1 + 100*a.getElectronicParams().getGain() + 0.5)
        # Mark the amplifier
        dataSec = a.getDataSec(isTrimmed)
        if a.getReadoutCorner() == cameraGeom.Amp.LLC:
            x, y = dataSec.getX0(),     dataSec.getY0()
        elif a.getReadoutCorner() == cameraGeom.Amp.LRC:
            x, y = dataSec.getX1() - 2, dataSec.getY0()
        elif a.getReadoutCorner() == cameraGeom.Amp.ULC:
            x, y = dataSec.getX0()    , dataSec.getY1() - 2
        elif a.getReadoutCorner() == cameraGeom.Amp.URC:
            x, y = dataSec.getX1() - 2, dataSec.getY1() - 2
        else:
            assert(not "Possible readoutCorner")

        ccdImage.Factory(ccdImage, afwImage.BBox(afwImage.PointI(x, y), 3, 3)).set(0)

    return ccdImage

def showCcd(ccd, ccdImage="", ccdOrigin=None, isTrimmed=None, frame=None, overlay=True):
    """Show a CCD on ds9.  If cameraImage isn't "", an image will be created based on the properties
of the detectors"""
    
    if isTrimmed is None:
        isTrimmed = ccd.isTrimmed()

    if ccdImage == "":
        ccdImage = makeImageFromCcd(ccd)

        title = ccd.getId().getName()
        if isTrimmed:
            title += "(trimmed)"
        ds9.mtv(ccdImage, frame=frame, title=title)

    if not overlay:
        return

    for a in ccd:
        displayUtils.drawBBox(a.getAllPixels(isTrimmed), origin=ccdOrigin, borderWidth=0.49, frame=frame)
        
        if not isTrimmed:
            displayUtils.drawBBox(a.getBiasSec(), origin=ccdOrigin,
                                  borderWidth=0.49, ctype=ds9.RED, frame=frame)
            displayUtils.drawBBox(a.getDataSec(), origin=ccdOrigin,
                                  borderWidth=0.49, ctype=ds9.BLUE, frame=frame)
        # Label each Amp
        ap = a.getAllPixels(isTrimmed)
        xc, yc = (ap.getX0() + ap.getX1())//2, (ap.getY0() + ap.getY1())//2
        cen = afwGeom.Point2I.makeXY(xc, yc)
        if ccdOrigin:
            xc += ccdOrigin[0]
            yc += ccdOrigin[1]

        ds9.dot(str(ccd.findAmp(cen).getId().getSerial()), xc, yc, frame=frame)

    displayUtils.drawBBox(ccd.getAllPixels(isTrimmed), origin=ccdOrigin,
                          borderWidth=0.49, ctype=ds9.MAGENTA, frame=frame)

def makeImageFromRaft(raft, filename="", raftCenter=None):
    """Make an Image of a Raft"""

    raftImage = afwImage.ImageU(raft.getAllPixels().getDimensions())

    for det in raft:
        ccd = cameraGeom.cast_Ccd(det)
        
        bbox = ccd.getAllPixels(True)
        origin = ccd.getCenterPixel() - \
                 afwGeom.Extent2I.makeXY(bbox.getWidth()/2, bbox.getHeight()/2)
        if raftCenter:
            origin = origin + afwGeom.Extent2I(raftCenter)

        bbox = ccd.getAllPixels(True).clone()
        bbox.shift(origin[0], origin[1])
        ccdImage = raftImage.Factory(raftImage, bbox)

        ccdImage <<= makeImageFromCcd(ccd, filename, isTrimmed=True)

    return raftImage

def showRaft(raft, raftImage="", raftOrigin=None, frame=None, overlay=True):
    """Show a Raft on ds9.  If cameraImage isn't "", an image will be created based on the
properties of the detectors"""

    raftCenter = afwGeom.Point2I.makeXY(raft.getAllPixels().getWidth()/2, raft.getAllPixels().getHeight()/2)
    if raftOrigin:
        raftCenter += afwGeom.Extent2I(raftOrigin)

    if isinstance(raftImage, str):
        raftImage = makeImageFromRaft(raft, filename=raftImage, raftCenter=raftCenter)

    if raftImage is not None:
        ds9.mtv(raftImage, frame=frame, title=raft.getId().getName())

    for det in raft:
        ccd = cameraGeom.cast_Ccd(det)
        
        bbox = ccd.getAllPixels(True)
        origin = ccd.getCenterPixel() - \
                 afwGeom.Extent2I.makeXY(bbox.getWidth()/2, bbox.getHeight()/2) + afwGeom.Extent2I(raftCenter)
            
        if overlay:
            if True:
                name = ccd.getId().getName()
            else:
                name = str(ccd.getCenter())
            ds9.dot(name, origin[0] + bbox.getWidth()/2, origin[1] + bbox.getHeight()/2, frame=frame)

        if raftImage is None:
            ccdImage = None
        else:
            bbox = ccd.getAllPixels(True).clone()
            bbox.shift(origin[0], origin[1])
            ccdImage = raftImage.Factory(raftImage, bbox)

        showCcd(ccd, ccdImage, isTrimmed=True, frame=frame, ccdOrigin=origin, overlay=overlay)

def makeImageFromCamera(camera, filename=""):
    """Make an Image of a Camera"""

    cameraImage = afwImage.ImageU(camera.getAllPixels().getDimensions())
    for det in camera:
        raft = cameraGeom.cast_Raft(det);
        bbox = raft.getAllPixels().clone()
        origin = camera.getCenterPixel() + afwGeom.Extent2I(raft.getCenterPixel()) - \
                 afwGeom.Extent2I.makeXY(bbox.getWidth()/2, bbox.getHeight()/2) 
        bbox.shift(origin[0], origin[1])
        im = cameraImage.Factory(cameraImage, bbox)

        im <<= makeImageFromRaft(raft, filename,
                                 afwGeom.Point2I.makeXY(bbox.getWidth()/2, bbox.getHeight()/2))
        im += raft.getId().getSerial()

    return cameraImage

def showCamera(camera, filename=None, frame=None, overlay=True):
    """Show a Camera on ds9.  If filename is provided an image will be created by reading it (see
makeImageFromCcd for details); if it is"", an image will be created based on the properties of the detectors"""

    if filename is None:
        cameraImage = None
    else:
        cameraImage = makeImageFromCamera(camera, filename)

    if cameraImage is not None:
        ds9.mtv(cameraImage, frame=frame, title=camera.getId().getName())

    for det in camera:
        raft = cameraGeom.cast_Raft(det)
        
        center = camera.getCenterPixel() + afwGeom.Extent2I(raft.getCenterPixel())
        if overlay:
            bbox = raft.getAllPixels()
            ds9.dot(raft.getId().getName(), center[0], center[1], frame=frame)

        showRaft(raft, None, frame=frame, overlay=overlay,
                 raftOrigin=center - afwGeom.Extent2I.makeXY(raft.getAllPixels().getWidth()/2,
                                                             raft.getAllPixels().getHeight()/2))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def describeRaft(raft, indent=""):
    """Describe an entire Raft"""
    descrip = []

    size = raft.getSize()
    descrip.append("%sRaft \"%s\",  %gx%g  BBox %s" % (indent, raft.getId(),
                                                        size[0], size[1], raft.getAllPixels()))
    
    for d in cameraGeom.cast_Raft(raft):
        cenPixel = d.getCenterPixel()
        cen = d.getCenter()

        descrip.append("%sCcd: %s (%5d, %5d) %s  (%7.1f, %7.1f)" % \
                       ((indent + "    "),
                        d.getAllPixels(True), cenPixel[0], cenPixel[1],
                        cameraGeom.ReadoutCorner(d.getOrientation().getNQuarter()), cen[0], cen[1]))
            
    return "\n".join(descrip)

def describeCamera(camera):
    """Describe an entire Camera"""
    descrip = []

    size = camera.getSize()
    descrip.append("Camera \"%s\",  %gx%g  BBox %s" % \
                   (camera.getId(), size[0], size[1], camera.getAllPixels()))

    for raft in camera:
        descrip.append(describeRaft(raft, "    "))

    return "\n".join(descrip)
