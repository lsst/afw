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

import lsst.pex.policy as pexPolicy
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
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
        self.isTrimmed = False

    def getImage(self, ccd, amp=None, imageFactory=afwImage.ImageU):
        """Return the image of the chip with cameraGeom.Id == id; if provided only read the given BBox"""

        return self.getImageFromFilename(self.imageFile, ccd, amp, imageFactory=imageFactory)

    def getImageFromFilename(self, fileName, ccd, amp=None, hdu=0, imageFactory=afwImage.ImageU,
                             oneAmpPerFile=False):
        """Return the image of the chip with cameraGeom.Id == id; if provided only read the given BBox"""

        if amp:
            if self.isTrimmed:
                bbox = amp.getDiskDataSec()
            else:
                bbox = amp.getDiskAllPixels()
        else:
            bbox = ccd.getAllPixels()

        md = None
        return imageFactory(fileName, hdu, md, bbox)

    def setTrimmed(self, doTrim):
        self.isTrimmed = doTrim

class SynthesizeCcdImage(GetCcdImage):
    """A class to return an Image of a given Ccd based on its cameraGeometry"""
    
    def __init__(self, isTrimmed=True):
        self.isTrimmed = isTrimmed

    def getImage(self, ccd, amp, imageFactory=afwImage.ImageU):
        """Return an image of the specified amp in the specified ccd"""
        
        bbox = amp.getAllPixels(self.isTrimmed)
        im = imageFactory(bbox.getDimensions())
        x0, y0 = bbox.getX0(), bbox.getY0()

        im += int(amp.getElectronicParams().getReadNoise())
        bbox = amp.getDataSec(self.isTrimmed).clone()
        bbox.shift(-x0, -y0)
        sim = imageFactory(im, bbox)
        sim += int(1 + 100*amp.getElectronicParams().getGain() + 0.5)
        # Mark the amplifier
        dataSec = amp.getDataSec(self.isTrimmed)
        if amp.getReadoutCorner() == cameraGeom.Amp.LLC:
            x, y = dataSec.getX0(),     dataSec.getY0()
        elif amp.getReadoutCorner() == cameraGeom.Amp.LRC:
            x, y = dataSec.getX1() - 2, dataSec.getY0()
        elif amp.getReadoutCorner() == cameraGeom.Amp.ULC:
            x, y = dataSec.getX0()    , dataSec.getY1() - 2
        elif amp.getReadoutCorner() == cameraGeom.Amp.URC:
            x, y = dataSec.getX1() - 2, dataSec.getY1() - 2
        else:
            assert(not "Possible readoutCorner")

        imageFactory(im, afwImage.BBox(afwImage.PointI(x - x0, y - y0), 3, 3)).set(0)

        return im

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def getGeomPolicy(cameraGeomPolicyFile):
    """Return a Policy describing a Camera's geometry"""
    
    if os.path.exists(cameraGeomPolicyFile):
        return pexPolicy.Policy(cameraGeomPolicyFile)

    policyFile = pexPolicy.DefaultPolicyFile("afw", "CameraGeomDictionary.paf", "policy")
    defPolicy = pexPolicy.Policy.createPolicy(policyFile, policyFile.getRepositoryPath(), True)

    policyFile = pexPolicy.DefaultPolicyFile("afw", cameraGeomPolicyFile, "examples")
    geomPolicy = pexPolicy.Policy.createPolicy(policyFile, policyFile.getRepositoryPath(), True)

    geomPolicy.mergeDefaults(defPolicy.getDictionary())

    return geomPolicy

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
    #
    # Find the proper electronic parameters.  The Ccd name may be specified as "*" to match all detectors,
    # a feature that's probably mostly useful for testing
    #
    electronicPol = geomPolicy.get("Electronic")
    electronics = {}
    for pol in electronicPol.getArray("Raft"):
        for pol in pol.getArray("Ccd"):
            electronicCcdName = pol.get("name")
            if electronicCcdName in ("*", ccdId.getName()):
                electronics["ccdName"] = electronicCcdName
                for p in pol.getArray("Amp"):
                    electronics[tuple(p.getArray("index"))] = p
                break
    #
    # Actually build the Ccd
    #
    ccd = cameraGeom.Ccd(ccdId, pixelSize)

    if nCol*nRow != len(ccdPol.getArray("Amp")):
        raise RuntimeError, ("Expected location of %d amplifiers, got %d" % \
                             (nCol*nRow, len(ccdPol.getArray("Amp"))))

    if ccdInfo is None:
        ampSerial = [0]
    else:
        ampSerial = ccdInfo.get("ampSerial", [0])
    ampSerial0 = None                   # used in testing
        
    readoutCorners = dict(LLC = cameraGeom.Amp.LLC,
                          LRC = cameraGeom.Amp.LRC,
                          ULC = cameraGeom.Amp.ULC,
                          URC = cameraGeom.Amp.URC)
    for ampPol in ccdPol.getArray("Amp"):
        if ampPol.exists("serial"):
            serial = ampPol.get("serial")
            ampSerial[0] = serial
        else:
            serial = ampSerial[0]
        ampSerial[0] += 1

        if ampSerial0 is None:
            ampSerial0 = serial

        Col, Row = index = tuple(ampPol.getArray("index"))
        c =  ampPol.get("readoutCorner")

        if Col not in range(nCol) or Row not in range(nRow):
            raise RuntimeError, ("Amp location %d, %d is not in 0..%d, 0..%d" % (Col, Row, nCol, nRow))

        try:
            ePol = electronics[index]
            gain = ePol.get("gain")
            readNoise = ePol.get("readNoise")
            saturationLevel = ePol.get("saturationLevel")
        except KeyError:
            if electronics.get("ccdName") != "*":
                raise RuntimeError, ("Unable to find electronic info for Ccd \"%s\", Amp %s" %
                                     (ccd.getId(), serial))
            gain, readNoise, saturationLevel = 0, 0, 0
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
        amp = cameraGeom.Amp(cameraGeom.Id(serial, "ID%d" % serial, Col, Row),
                             allPixels, biasSec, dataSec, c, eParams)
        #
        # Actually add amp to the Ccd
        #
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
    #
    # Discover how the per-amp data is laid out on disk; it's common for the data acquisition system to
    # put together a single image for an entire CCD, but it isn't mandatory
    #
    diskFormatPol = geomPolicy.get("CcdDiskLayout")
    hduPerAmp = diskFormatPol.get("HduPerAmp")
    ampDiskLayout = {}
    if hduPerAmp:
        for p in diskFormatPol.getArray("Amp"):
            ampDiskLayout[p.get("serial")] = (p.get("hdu"), p.get("flipLR"), p.get("flipTB"))
    #
    # Build the Raft
    #
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

        raft.addDetector(afwGeom.makePointI(Col, Row),
                         afwGeom.makePointD(xc, yc),
                         cameraGeom.Orientation(nQuarter, pitch, roll, yaw), ccd)

        #
        # Set the on-disk layout parameters now that we've possibly rotated the Ccd to fit the raft
        #
        if hduPerAmp:
            for amp in ccd:
                hdu, flipLR, flipTB = ampDiskLayout[amp.getId().getSerial()]
                amp.setDiskLayout(afwGeom.makePointI(amp.getAllPixels().getX0(), amp.getAllPixels().getY0()),
                                  nQuarter, flipLR, flipTB)

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
        camera.addDetector(afwGeom.makePointI(Col, Row),
                           afwGeom.makePointD(xc, yc), cameraGeom.Orientation(), raft)

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

def makeAmpImageFromCcd(amp, imageSource=SynthesizeCcdImage(), isTrimmed=None, imageFactory=afwImage.ImageU):
    """Make an Image of an Amp"""

    return imageSource.getImage(amp, imageFactory=imageFactory)

def makeImageFromCcd(ccd, imageSource=SynthesizeCcdImage(), amp=None,
                     isTrimmed=None, imageFactory=afwImage.ImageU):
    """Make an Image of a Ccd (or just a single amp)"""

    if isTrimmed is None:
        isTrimmed = ccd.isTrimmed()
    imageSource.setTrimmed(isTrimmed)

    if amp:
        ampImage = imageFactory(amp.getAllPixels(isTrimmed).getDimensions())
        ampImage <<= imageSource.getImage(ccd, amp, imageFactory=imageFactory)

        return ampImage

    ccdImage = imageFactory(ccd.getAllPixels(isTrimmed).getDimensions())
        
    for a in ccd:
        im = ccdImage.Factory(ccdImage, a.getAllPixels(isTrimmed))
        im <<= a.prepareAmpData(imageSource.getImage(ccd, a, imageFactory=imageFactory))

    return ccdImage

def showCcd(ccd, ccdImage="", amp=None, ccdOrigin=None, isTrimmed=None, frame=None, overlay=True):
    """Show a CCD on ds9.  If cameraImage is "", an image will be created based on the properties
of the detectors"""
    
    if isTrimmed is None:
        isTrimmed = ccd.isTrimmed()

    if ccdImage == "":
        ccdImage = makeImageFromCcd(ccd)

    if ccdImage:
        title = ccd.getId().getName()
        if amp:
            title += ":%d" % amp.getId().getSerial()
        if isTrimmed:
            title += "(trimmed)"
        ds9.mtv(ccdImage, frame=frame, title=title)

    if not overlay:
        return

    if amp:
        bboxes = [(amp.getAllPixels(isTrimmed), 0.49, None),]
        x0, y0 = bboxes[0][0].getLLC()
        if not isTrimmed:
            bboxes.append((amp.getBiasSec(), 0.49, ds9.RED)) 
            bboxes.append((amp.getDataSec(), 0.49, ds9.BLUE))

        for bbox, borderWidth, ctype in bboxes:
            bbox = bbox.clone()
            bbox.shift(-x0, -y0)
            displayUtils.drawBBox(bbox, borderWidth=borderWidth, ctype=ctype, frame=frame)

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
        cen = afwGeom.makePointI(xc, yc)
        if ccdOrigin:
            xc += ccdOrigin[0]
            yc += ccdOrigin[1]

        ds9.dot(str(ccd.findAmp(cen).getId().getSerial()), xc, yc, frame=frame)

    displayUtils.drawBBox(ccd.getAllPixels(isTrimmed), origin=ccdOrigin,
                          borderWidth=0.49, ctype=ds9.MAGENTA, frame=frame)

def makeImageFromRaft(raft, imageSource=SynthesizeCcdImage(), raftCenter=None, imageFactory=afwImage.ImageU):
    """Make an Image of a Raft"""

    raftImage = imageFactory(raft.getAllPixels().getDimensions())

    for det in raft:
        ccd = cameraGeom.cast_Ccd(det)
        
        bbox = ccd.getAllPixels(True)
        origin = ccd.getCenterPixel() - \
                 afwGeom.makeExtentI(bbox.getWidth()/2, bbox.getHeight()/2)
        if raftCenter:
            origin = origin + afwGeom.Extent2I(raftCenter)

        bbox = ccd.getAllPixels(True).clone()
        bbox.shift(origin[0], origin[1])
        ccdImage = raftImage.Factory(raftImage, bbox)
            
        ccdImage <<= makeImageFromCcd(ccd, imageSource, isTrimmed=True)

    return raftImage

def showRaft(raft, imageSource=SynthesizeCcdImage(), raftOrigin=None, frame=None, overlay=True):
    """Show a Raft on ds9.

If imageSource isn't None, an image using the images specified by imageSource"""

    raftCenter = afwGeom.makePointI(raft.getAllPixels().getWidth()/2, raft.getAllPixels().getHeight()/2)
    if raftOrigin:
        raftCenter += afwGeom.Extent2I(raftOrigin)

    if imageSource is None:
        raftImage = None
    else:
        raftImage = makeImageFromRaft(raft, imageSource=imageSource, raftCenter=raftCenter)

    if raftImage:
        ds9.mtv(raftImage, frame=frame, title=raft.getId().getName())

    if not raftImage and not overlay:
        return

    for det in raft:
        ccd = cameraGeom.cast_Ccd(det)
        
        bbox = ccd.getAllPixels(True)
        origin = ccd.getCenterPixel() - \
                 afwGeom.makeExtentI(bbox.getWidth()/2, bbox.getHeight()/2) + afwGeom.Extent2I(raftCenter)
            
        if True:
            name = ccd.getId().getName()
        else:
            name = str(ccd.getCenter())
        ds9.dot(name, origin[0] + bbox.getWidth()/2, origin[1] + bbox.getHeight()/2, frame=frame)

        showCcd(ccd, None, isTrimmed=True, frame=frame, ccdOrigin=origin, overlay=overlay)

def makeImageFromCamera(camera, imageSource=None, imageFactory=afwImage.ImageU):
    """Make an Image of a Camera"""

    cameraImage = imageFactory(camera.getAllPixels().getDimensions())
    for det in camera:
        raft = cameraGeom.cast_Raft(det);
        bbox = raft.getAllPixels().clone()
        origin = camera.getCenterPixel() + afwGeom.Extent2I(raft.getCenterPixel()) - \
                 afwGeom.makeExtentI(bbox.getWidth()/2, bbox.getHeight()/2) 
        bbox.shift(origin[0], origin[1])
        im = cameraImage.Factory(cameraImage, bbox)

        im <<= makeImageFromRaft(raft, imageSource,
                                 afwGeom.makePointI(bbox.getWidth()/2, bbox.getHeight()/2))
        im += raft.getId().getSerial()

    return cameraImage

def showCamera(camera, imageSource=SynthesizeCcdImage(), frame=None, overlay=True):
    """Show a Camera on ds9 (with the specified frame); if overlay show the IDs and amplifier boundaries

If imageSource is provided its getImage method will be called to return a CCD image (e.g. a
cameraGeom.GetCcdImage object); if it is "", an image will be created based on the properties
of the detectors"""

    if imageSource is None:
        cameraImage = None
    else:
        cameraImage = makeImageFromCamera(camera, imageSource)

    if cameraImage:
        ds9.mtv(cameraImage, frame=frame, title=camera.getId().getName())

    for det in camera:
        raft = cameraGeom.cast_Raft(det)
        
        center = camera.getCenterPixel() + afwGeom.Extent2I(raft.getCenterPixel())

        if overlay:
            bbox = raft.getAllPixels()
            ds9.dot(raft.getId().getName(), center[0], center[1], frame=frame)

        showRaft(raft, None, frame=frame, overlay=overlay,
                 raftOrigin=center - afwGeom.makeExtentI(raft.getAllPixels().getWidth()/2,
                                                         raft.getAllPixels().getHeight()/2))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def showMosaic(fileName, geomPolicy=None, camera=None,
               display=True, what=cameraGeom.Camera, id=None, overlay=False, describe=True, doTrim=False,
               imageFactory=afwImage.ImageU, frame=None):
    """Show a mosaic built from the MEF imageFile containing an exposure

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

        if display:
            ampImage = makeImageFromCcd(ccd, imageSource, amp=amp, imageFactory=imageFactory)
            showCcd(ccd, ampImage, amp=amp, overlay=overlay, frame=frame)
    elif what == cameraGeom.Ccd:
        if id is None:
            ccd = makeCcd(geomPolicy)
        else:
            ccd = findCcd(camera, id)

        if not ccd:
            raise RuntimeError, "Failed to find Ccd %s" % id

        ccd.setTrimmed(doTrim)

        if display:
            ccdImage = makeImageFromCcd(ccd, imageSource, imageFactory=imageFactory)
            showCcd(ccd, ccdImage, overlay=overlay, frame=frame)
    elif what == cameraGeom.Raft:
        if id:
            raft = findRaft(camera, id)
        else:
            raft = makeRaft(geomPolicy)
        if not raft:
            raise RuntimeError, "Failed to find Raft %s" % id

        #raft = makeRaft(geomPolicy, raftId=id)

        if display:
            showRaft(raft, imageSource, overlay=overlay, frame=frame)

        if describe:
            print describeRaft(raft)
    elif what == cameraGeom.Camera:
        if display:
            showCamera(camera, imageSource, overlay=overlay, frame=frame)

        if describe:
            print describeCamera(camera)
    else:
        raise RuntimeError, ("I don't know how to display %s" % what)

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

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def findAmp(parent, ccdId, ix, iy):
    """Find the Amp with the specified Id within the composite"""

    ccd = findCcd(parent, ccdId)
    for amp in ccd:
        if amp.getId().getIndex() == (ix, iy):
            return amp

    return None

def findCcd(parent, id):
    """Find the Ccd with the specified Id within the composite"""

    if isinstance(parent, cameraGeom.Camera):
        for d in parent:
            ccd = findCcd(cameraGeom.cast_Raft(d), id)
            if ccd:
                return ccd
    elif isinstance(parent, cameraGeom.Raft):
        d = parent.findDetector(id)
        if d:
            return cameraGeom.cast_Ccd(d)
    else:
        if parent.getId() == id:
            return cameraGeom.cast_Ccd(parent)
        
    return None

def findRaft(parent, id):
    """Find the Raft with the specified Id within the composite"""

    if isinstance(parent, cameraGeom.Camera):
        d = parent.findDetector(id)
        if d:
            return cameraGeom.cast_Raft(d)
    else:
        if parent.getId() == id:
            return raft

    return None

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def makeDefects(geomPolicy):
    """Create a dictionary of DefectSets from a pexPolicy::Policy

The dictionay is indexed by an Id object --- remember to compare by str(id) not object identity
    """

    defectsDict = {}
    defectListPol = geomPolicy.get("Defects")
    for raftPol in defectListPol.getArray("Raft"):
        for defectPol in raftPol.getArray("Ccd"):
            defects = afwImage.DefectSet()
            ccdId = cameraGeom.Id(defectPol.get("serial"), defectPol.get("name"))
            defectsDict[ccdId] = defects

            for defect in defectPol.getArray("Defect"):
                x0 = defect.get("x0")
                y0 = defect.get("y0")

                x1 = y1 = width = height = None
                if defect.exists("x1"):
                    x1 = defect.get("x1")
                if defect.exists("y1"):
                    y1 = defect.get("y1")
                if defect.exists("width"):
                    width = defect.get("width")
                if defect.exists("height"):
                    height = defect.get("height")

                if x1 is None:
                    if width:
                        x1 = x0 + width - 1
                    else:
                        raise RuntimeError, ("Defect at (%d,%d) for CCD (%s) has no x1/width" % (x0, y0, ccdId))
                else:
                    if width:
                        if x1 != x0 + width - 1:
                            raise RuntimeError, \
                                  ("Defect at (%d,%d) for CCD (%s) has inconsistent x1/width = %d,%d" % \
                                   (x0, y0, ccdId, x1, width))

                if y1 is None:
                    if height:
                        y1 = y0 + height - 1
                    else:
                        raise RuntimeError, ("Defect at (%d,%d) for CCD (%s) has no y1/height" % (x0, y0, ccdId))
                else:
                    if height:
                        if y1 != y0 + height - 1:
                            raise RuntimeError, \
                                  ("Defect at (%d,%d) for CCD (%s) has inconsistent y1/height = %d,%d" % \
                                   (x0, y0, ccdId, y1, height))

                bbox = afwImage.BBox(afwImage.PointI(x0, y0), afwImage.PointI(x1, y1))
                defects.push_back(afwImage.DefectBase(bbox))

        return defectsDict

