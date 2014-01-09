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

import math
import multiprocessing
import os
import re
import sys
import unittest
import numpy as np
try:
    import pyfits
except ImportError:
    pyfits = None

import lsst.daf.persistence as dafPersist
import lsst.pex.policy as pexPolicy
import lsst.pex.logging as pexLog
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
    force = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def trimRawCallback(im, ccd=None, butler=None, imageSource=None, correctGain=False,
                    subtractDark=False, subtractBias=False,
                    convertToFloat=False):
    """A callback function that subtracts the bias and trims a raw image"""
    if ccd is None:
        ccd = cameraGeom.cast_Ccd(im.getDetector())
    if hasattr(im, "getMaskedImage"):
        im = im.getMaskedImage()
    if convertToFloat and hasattr(im, "convertF"):
        im = im.convertF()

    ccdImage = im.Factory(ccd.getAllPixels(True))

    for a in ccd:
        if False:
            im[a.getDiskDataSec()] -= \
                afwMath.makeStatistics(im[a.getDiskBiasSec()], afwMath.MEDIAN).getValue()
        else:
            biasSec = a.getDiskBiasSec()
            width, height = biasSec.getDimensions()

            vals = []
            for i in range(2):
                vals.append(afwMath.makeStatistics(
                        im[afwGeom.BoxI(
                                afwGeom.PointI(biasSec.getMinX(), biasSec.getMinY() + i*(height//2)),
                                afwGeom.ExtentI(width, height//2))],
                        afwMath.MEANCLIP).getValue())

            dataSec = a.getDiskDataSec()
            try:
                ima = im.getImage().getArray()
            except:
                ima = im.getArray()

            if False:
                ramp = np.arange(im.getHeight())
                ramp = 0.5*(vals[0] + vals[1]) + \
                    (ramp - biasSec.getMinY() - 0.5*height)*2*(vals[1] - vals[0])/height
                
                for i in range(dataSec.getMinX(), dataSec.getMinX() + dataSec.getWidth()):
                    ima[:, i] -= ramp # n.b. numpy indexes are (y, x) not our (x, y)
            elif False:
                X, Y = np.meshgrid(np.arange(im.getWidth()), np.arange(im.getHeight()))
                
                ramp = 0.5*(vals[0] + vals[1]) + \
                    (Y - biasSec.getMinY() - 0.5*height)*2*(vals[1] - vals[0])/height
                
                ima[:] -= ramp
            else:                       # do a per-row bias subtraction
                y0, y1 = a.getDiskDataSec().getMinY(), a.getDiskDataSec().getMaxY()
                for y in range(y0, y1 + 1):
                    im[:, y] -= afwMath.makeStatistics(im[a.getDiskBiasSec()][:, y - y0],
                                                       afwMath.MEANCLIP).getValue()
        a.setTrimmed(True)
        
        ccdImage[a.getAllPixels()] <<= a.prepareAmpData(im[a.getDiskDataSec()])

    if subtractDark or subtractBias:
        assert imageSource
        if not butler:
            butler = imageSource.butler

        md = butler.get("raw_md", ccd=ccd.getId().getSerial(), **imageSource.kwargs)
        if subtractDark:
            exptime = afwImage.Calib(md).getExptime()
            dark = butler.get("dark", ccd=ccd.getId().getSerial(), **imageSource.kwargs)
            dark = dark.getMaskedImage().getImage()
            dark *= exptime

            ccdImage -= dark
        if subtractBias:
            bias = butler.get("bias", ccd=ccd.getId().getSerial(), **imageSource.kwargs)
            bias = bias.getMaskedImage().getImage()

            ccdImage -= bias
    #
    # Convert from DN to e?
    #
    if correctGain:
        for a in ccd:
            ccdImage[a.getAllPixels()] *= a.getElectronicParams().getGain()

    return ccdImage
    
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class GetCcdImage(object):
    """A class to return an Image of a given Ccd"""

    def __init__(self, imageFile=None):
        self.imageFile = imageFile
        self.isTrimmed = False
        self.imageIsBinned = False    # i.e. images returned by getImage aren't already binned
        self.isRaw = True
        self.filter = None

    def getImage(self, ccd, amp=None, imageFactory=afwImage.ImageU):
        """Return the image of the chip with cameraGeom.Id == id; if provided only read the given BBox"""

        return self.getImageFromFilename(self.imageFile, ccd, amp, imageFactory=imageFactory)

    def getImageFromFilename(self, fileName, ccd, amp=None, hdu=0, imageFactory=afwImage.ImageU,
                             oneAmpPerFile=False):
        """Return the image of the chip with cameraGeom.Id == id; if provided only read the given BBox"""

        if amp:
            if self.isTrimmed:
                bbox = amp.getElectronicDataSec()
            else:
                bbox = amp.getElectronicAllPixels()
        else:
            bbox = ccd.getAllPixels()

        md = None
        return imageFactory(fileName, hdu, md, bbox)

    def setTrimmed(self, doTrim):
        self.isTrimmed = doTrim

class ButlerImage(GetCcdImage):
    """A class to return an Image of a given Ccd based on its cameraGeometry"""
    
    def __init__(self, butler=None, type="raw", isTrimmed=True, callback=None,
                 gravity=None, background=np.nan, verbose=False, *args, **kwargs):
        """Initialise
        gravity  If the image returned by the butler is trimmed (e.g. some of the SuprimeCam CCDs)
                 Specify how to fit the image into the available space; N => align top, W => align left
        background  The value of any pixels that lie outside the CCDs
        callback  A function called with (image, Ccd, butler) for every image, which returns the image
                  to be displayed (e.g. trimRawCallback)
        """
        super(ButlerImage, self).__init__(*args)
        self.isTrimmed = isTrimmed
        self.type = type
        self.butler = butler
        self.verbose = verbose
        self.kwargs = kwargs
        self.isRaw = False
        self.gravity = gravity
        self.background = background
        self.callback = callback

    def getImage(self, ccd, amp=None, imageFactory=afwImage.ImageU):
        """Return an image of the specified amp in the specified ccd"""

        im = None
        if self.butler is not None:
            ccdNo = ccd.getId().getSerial()
            try:
                im = self.butler.get(self.type, ccd=ccdNo, immediate=True, **self.kwargs)
                try:
                    ccd = cameraGeom.cast_Ccd(im.getDetector()) # ccd may be a PyCcd
                except Exception, e:
                    print "Failed to update Ccd:", e
                try:
                    self.filter = im.getFilter().getName()
                except:
                    raise
                im = im.getMaskedImage()

                if not re.search(r"MaskedImage", str(imageFactory)):
                    im = im.getImage()
            except Exception, e:
                if self.verbose:
                    print "Butler failed: %s" % e

        if im is None:
            return imageFactory(*ccd.getAllPixels(self.isTrimmed).getDimensions())
                
        if self.type == "raw":
            im = im.convertF()

        if not self.callback:
            return im

        try:
            return self.callback(im, ccd, self.butler, imageSource=self)
        except Exception, e:
            if self.verbose:
                print "callback failed: %s" % e
            return imageFactory(*ccd.getAllPixels(self.isTrimmed).getDimensions())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def mergeGeomDefaults(cameraGeomPolicy):
   policyFile = pexPolicy.DefaultPolicyFile("afw", "CameraGeomDictionary.paf", "policy")
   defPolicy = pexPolicy.Policy.createPolicy(policyFile, policyFile.getRepositoryPath(), True)

   cameraGeomPolicy.mergeDefaults(defPolicy.getDictionary())
   
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def getGeomPolicy(cameraGeomPolicy):
    """Return a Policy describing a Camera's geometry given a filename; the Policy will be validated using the
    dictionary, and defaults will be supplied.  If you pass a Policy, it will be validated and completed.
"""

    policyFile = pexPolicy.DefaultPolicyFile("afw", "CameraGeomDictionary.paf", "policy")
    defPolicy = pexPolicy.Policy.createPolicy(policyFile, policyFile.getRepositoryPath(), True)

    if isinstance(cameraGeomPolicy, pexPolicy.Policy):
        geomPolicy = cameraGeomPolicy
    else:
        if os.path.exists(cameraGeomPolicy):
            geomPolicy = pexPolicy.Policy.createPolicy(cameraGeomPolicy)
        else:
            policyFile = pexPolicy.DefaultPolicyFile("afw", cameraGeomPolicy, "examples")
            geomPolicy = pexPolicy.Policy.createPolicy(policyFile, policyFile.getRepositoryPath(), True)

    geomPolicy.mergeDefaults(defPolicy.getDictionary())

    return geomPolicy

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def makeLinearityFromPolicy(linPol, gain=1.0):
    """Make and return a Linearity object from a suitable policy"""
    assert linPol.get("type") == "PROPORTIONAL", "Checked in CameraGeomDictionary.paf"

    threshold = linPol.get("threshold")
    maxCorrectable = linPol.get("maxCorrectable")
    coefficient = linPol.get("coefficient")

    if linPol.get("intensityUnits") == "ELECTRONS":
        threshold = int(threshold/gain + 0.5)
        maxCorrectable = int(maxCorrectable/gain + 0.5)
        coefficient /= gain
    else:
        assert linPol.get("intensityUnits") == "DN", "Checked in CameraGeomDictionary.paf"

    return cameraGeom.Linearity(cameraGeom.Linearity.PROPORTIONAL,
                                threshold, maxCorrectable, coefficient)


def makeCcd(geomPolicy, ccdId=None, ccdInfo=None, defectDict={}, ccdDescription=None):
    """Build a Ccd from a set of amplifiers given a suitable pex::Policy

If ccdInfo is provided it's set to various facts about the CCDs which are used in unit tests.  Note
in particular that it has an entry ampSerial which is a single-element list, the amplifier serial counter
    """
    ccdPol = None
    if ccdDescription:
        try:
            tId = cameraGeom.Id(ccdDescription.get("serial"), ccdDescription.get("name"))
        except Exception, e:
            tId = cameraGeom.Id(0, "unknown")
        if ccdId:
            if not tId == ccdId:
                raise("Identifiers don't agree -- Passed: %s, Policy: %s"%(ccdId.__str__(), tId.__str__()))
        else:
            ccdId = tId
        ccdTempArr = geomPolicy.getArray("Ccd")
        for ct in ccdTempArr:
            if ct.get("ptype") == ccdDescription.get("ptype"):
                ccdPol = ct
    else:        
        ccdPol = geomPolicy.get("Ccd")
    if not ccdPol:
        raise("No valid CCD policy found")
    pixelSize = ccdPol.get("pixelSize")
    offsetUnit = ccdPol.get('offsetUnit')
    
    nCol = ccdPol.get("nCol")
    nRow = ccdPol.get("nRow")
    ccdType = ccdPol.get("ptype")
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
            electronicCcdType = pol.get("ptype")
            if electronicCcdName in ("*", ccdId.getName()) and electronicCcdType == ccdType:
                electronics["ccdName"] = electronicCcdName
                for p in pol.getArray("Amp"):
                    electronics[tuple(p.getArray("index"))] = p
                break
    #
    # Actually build the Ccd
    #
    ccd = cameraGeom.Ccd(ccdId, pixelSize)
    for k in defectDict.keys():
        if ccdId == k:
            ccd.setDefects(defectDict[k])
        else:
            pass

    if nCol*nRow != len(ccdPol.getArray("Amp")):
        msg = "Expected location of %d amplifiers, got %d" % (nCol*nRow, len(ccdPol.getArray("Amp")))
        
        if force:
            print >> sys.stderr, msg
        else:
            raise RuntimeError, msg

    if ccdInfo is None:
        ampSerial = [0]
    else:
        ampSerial = ccdInfo.get("ampSerial", [0])
    ampSerial0 = None                   # used in testing
        
    readoutCorners = dict(LLC = cameraGeom.Amp.LLC,
                          LRC = cameraGeom.Amp.LRC,
                          ULC = cameraGeom.Amp.ULC,
                          URC = cameraGeom.Amp.URC)

    diskCoordSys = dict(AMP = cameraGeom.Amp.AMP,
                        SENSOR = cameraGeom.Amp.SENSOR,
                        CAMERA = cameraGeom.Amp.CAMERA)

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
        ampType = ampPol.get("ptype")
        flipLR = ampPol.get("flipLR")
        try:
            coordSys = diskCoordSys[ampPol.get("diskCoordSys").upper()]        
          
        except Exception, e:
            raise("Coordinate system specified in the policy must be one of: %s"%(",".join(diskCoordSys.keys())))
        nQuarterAmp = ampPol.get("nQuarter")
        hdu = ampPol.get("hdu")

        if Col not in range(nCol) or Row not in range(nRow):
            msg = "Amp location %d, %d is not in 0..%d, 0..%d" % (Col, Row, nCol, nRow)
            if force:
                print >> sys.stderr, msg
                continue
            else:
                raise RuntimeError, msg

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
        ampPolArr = geomPolicy.getArray("Amp")
        ampPol = None
        for p in ampPolArr:
            if p.get("ptype") == ampType:
                ampPol = p
        if ampPol is None:
            raise RuntimeError, ("Unable to find bounding box info for Amp: %i, %i in Ccd \"%s\""%(Col, Row, ccd.getId()))

        minx, miny, maxx, maxy = tuple(ampPol.getArray("datasec"))
        dataSec = afwGeom.Box2I(afwGeom.Point2I(minx, miny), afwGeom.Point2I(maxx, maxy))
        minx, miny, maxx, maxy = tuple(ampPol.getArray("biassec"))
        biasSec = afwGeom.Box2I(afwGeom.Point2I(minx, miny), afwGeom.Point2I(maxx, maxy))
        eWidth = ampPol.get("ewidth")
        eHeight = ampPol.get("eheight")
        allPixelsInAmp = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(eWidth, eHeight))

        eParams = cameraGeom.ElectronicParams(gain, readNoise, saturationLevel)
        amp = cameraGeom.Amp(cameraGeom.Id(serial, "ID%d" % serial, Col, Row),
                             allPixelsInAmp, biasSec, dataSec, eParams)
        #The following maps how the amp pixels must be changed to go from electronic (on disk) coordinates
        #to detector coordinates.  This also sets the readout corner.
        amp.setElectronicToChipLayout(afwGeom.Point2I(Col, Row), nQuarterAmp, flipLR, coordSys)
        #
        # Actually add amp to the Ccd
        #
        ccd.addAmp(amp)
    #
    # Read any linearity information
    #
    # Start by looking for our CCD, then (failing that) for the default, with CCD serial number -1, 
    #
    def findLinearity(serial):
        pol = [pol for pol in geomPolicy.get("Linearity").getArray("Ccd") if pol.get("serial") == serial]
        if not pol:
            return None
        
        if len(pol) != 1:
            raise RuntimeError("Found more than one Linearity for serial %d" % (serial))
        return pol[0]

    linearityPolicy = findLinearity(ccdId.getSerial())
    if not linearityPolicy:
        linearityPolicy = findLinearity(-1)
    #
    # If we have a policy, set the linearity parameters
    #
    if linearityPolicy:
        linPols = {}
        for policy in linearityPolicy.getArray("Amp"):
            linPols[policy.get("serial")] = policy

        for amp in ccd:
            ep = amp.getElectronicParams()
            try:
                lin = makeLinearityFromPolicy(linPols[amp.getId().getSerial()], gain=ep.getGain())
            except KeyError:
                continue
            ep.setLinearity(lin)
    #
    # Information for the test code
    #
    if ccdInfo is not None:
        width, height = tuple(dataSec.getDimensions())
        ccdInfo.clear()
        ccdInfo["ampSerial"] = ampSerial
        ccdInfo["name"] = ccd.getId().getName()
        ccdInfo["ampWidth"], ccdInfo["ampHeight"] = width, height
        ccdInfo["width"], ccdInfo["height"] = nCol*eWidth, nRow*eHeight
        ccdInfo["trimmedWidth"], ccdInfo["trimmedHeight"] = nCol*width, nRow*height
        ccdInfo["pixelSize"] = pixelSize
        ccdInfo["offsetUnit"] = offsetUnit
        ccdInfo["ampIdMin"] = ampSerial0
        ccdInfo["ampIdMax"] = ampSerial[0] - 1

    return ccd

def makeRaft(geomPolicy, raftId=None, raftInfo=None, defectDict={}):
    """Build a Raft from a set of CCDs given a suitable pex::Policy
    
If raftInfo is provided it's set to various facts about the Rafts which are used in unit tests.  Note in
particular that it has an entry ampSerial which is a single-element list, the amplifier serial counter
"""

    if raftInfo is None:
        ccdInfo = {}
    else:
        ccdInfo = {"ampSerial" : raftInfo.get("ampSerial", [0])}

    if raftId and geomPolicy.isArray("Raft"):
        raftPol = None
        for p in geomPolicy.getArray("Raft"):
            if p.exists("name"):        # Build an Id from available information
                if p.exists("serial"):
                    rid = cameraGeom.Id(p.get("serial"), p.get("name"))
                else:
                    rid = cameraGeom.Id(p.get("name"))
            elif p.exists("serial"):
                rid = cameraGeom.Id(p.get("serial"))
            else:
                raise RuntimeError, "Please provide a raft name, a raft serial, or both"
                    
            if rid == raftId:
                raftPol = p
                break

        if not raftPol:
            raise RuntimeError, ("I can't find Raft %s" % raftId)            
    else:
        raftPol = geomPolicy.get("Raft")
        
    nCol = raftPol.get("nCol")
    nRow = raftPol.get("nRow")
    if not raftId:
        try:
            raftId = cameraGeom.Id(raftPol.get("serial"), raftPol.get("name"))
        except Exception, e:
            raftId = cameraGeom.Id(0, "unknown")
    #
    # Build the Raft
    #
    raft = cameraGeom.Raft(raftId, nCol, nRow)

    if nCol*nRow != len(raftPol.getArray("Ccd")):
        if False:                       # many cameras don't use filled rafts at the edge (e.g. HSC)
            msg = "Expected location of %d Ccds, got %d" % (nCol*nRow, len(raftPol.getArray("Ccd")))
            
            if force:
                print >> sys.stderr, msg
            else:
                raise RuntimeError, msg

    for ccdPol in raftPol.getArray("Ccd"):
        Col, Row   = ccdPol.getArray("index")
        xc, yc     = ccdPol.getArray("offset")
        
        nQuarter = ccdPol.get("nQuarter")
        pitch, roll, yaw = [afwGeom.Angle(a, afwGeom.degrees) for a in ccdPol.getArray("orientation")]

        if Col not in range(nCol) or Row not in range(nRow):
            msg = "Ccd location %d, %d is not in 0..%d, 0..%d" % (Col, Row, nCol, nRow)
            if force:
                print >> sys.stderr, msg
                continue
            else:
                raise RuntimeError, msg

        ccdId = cameraGeom.Id(ccdPol.get("serial"), ccdPol.get("name"))
        ccd = makeCcd(geomPolicy, ccdDescription=ccdPol, ccdId=ccdId, ccdInfo=ccdInfo, defectDict=defectDict)
        pixelSize = ccd.getPixelSize()

        offsetUnit = ccdInfo["offsetUnit"]
        if offsetUnit == 'pixels':
            xc *= pixelSize
            yc *= pixelSize
        
        raft.addDetector(afwGeom.Point2I(Col, Row),
                         cameraGeom.FpPoint(xc, yc),
                         cameraGeom.Orientation(nQuarter, pitch, roll, yaw), ccd)

        if raftInfo is not None:
            # Guess the gutter between detectors
            if (Col, Row) == (0, 0):
                xGutter, yGutter = xc, yc
            elif (Col, Row) == (nCol - 1, nRow - 1):
                if nCol == 1:
                    xGutter = 0.0
                else:
                    xGutter = (xc - xGutter)/float(nCol - 1) - ccd.getSize().getMm()[0]

                if nRow == 1:
                    yGutter = 0.0
                else:
                    yGutter = (yc - yGutter)/float(nRow - 1) - ccd.getSize().getMm()[1]

    if raftInfo is not None:
        raftInfo.clear()
        raftInfo["ampSerial"] = ccdInfo["ampSerial"]
        raftInfo["name"] = raft.getId().getName()
        raftInfo["pixelSize"] = ccd.getPixelSize()
        raftInfo["width"] =  nCol*ccd.getAllPixels(True).getWidth()
        raftInfo["height"] = nRow*ccd.getAllPixels(True).getHeight()
        raftInfo["widthMm"] =  nCol*ccd.getSize().getMm()[0] + (nCol - 1)*xGutter
        raftInfo["heightMm"] = nRow*ccd.getSize().getMm()[1] + (nRow - 1)*yGutter

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
   
    if geomPolicy.get("Defects").get("Raft").get("Ccd").isPolicy("Defect"):
        defDict = makeDefects(geomPolicy)
    else:
        defDict = {}

    for raftPol in cameraPol.getArray("Raft"):
        Col, Row = raftPol.getArray("index")
        xc, yc = raftPol.getArray("offset")
        raftId = cameraGeom.Id(raftPol.get("serial"), raftPol.get("name"))
        raft = makeRaft(geomPolicy, raftId, raftInfo, defectDict=defDict)
        camera.addDetector(afwGeom.Point2I(Col, Row),
                           cameraGeom.FpPoint(afwGeom.Point2D(xc, yc)),
                           cameraGeom.Orientation(), raft)

        if cameraInfo is not None:
            # Guess the gutter between detectors
            if (Col, Row) == (0, 0):
                xGutter, yGutter = xc, yc
            elif (Col, Row) == (nCol - 1, nRow - 1):
                if nCol == 1:
                    xGutter = 0.0
                else:
                    xGutter = (xc - xGutter)/float(nCol - 1) - raft.getSize().getMm()[0]
                    
                if nRow == 1:
                    yGutter = 0.0
                else:
                    yGutter = (yc - yGutter)/float(nRow - 1) - raft.getSize().getMm()[1]


    #######################
    # get the distortion
    distortPolicy = cameraPol.get('Distortion')
    distortActive = distortPolicy.get('active')
    activePolicy  = distortPolicy.get(distortActive)

    distort = None
    if distortActive == "NullDistortion":
        distort = cameraGeom.NullDistortion()
    elif distortActive == "RadialPolyDistortion":
        coeffs = activePolicy.getArray('coeffs')
        coefficientsDistort = activePolicy.get('coefficientsDistort')
        distort = cameraGeom.RadialPolyDistortion(coeffs, coefficientsDistort)
    camera.setDistortion(distort)


    if cameraInfo is not None:
        cameraInfo.clear()
        cameraInfo["ampSerial"] = raftInfo["ampSerial"]
        cameraInfo["name"] = camera.getId().getName()
        cameraInfo["width"] =  nCol*raft.getAllPixels().getWidth()
        cameraInfo["height"] = nRow*raft.getAllPixels().getHeight()
        cameraInfo["pixelSize"] = raft.getPixelSize()
        cameraInfo["widthMm"] =  nCol*raft.getSize().getMm()[0] + (nCol - 1)*xGutter
        cameraInfo["heightMm"] = nRow*raft.getSize().getMm()[1] + (nRow - 1)*yGutter

    return camera

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def makeAmpImageFromCcd(amp, imageSource=ButlerImage(), isTrimmed=None, imageFactory=afwImage.ImageU):
    """Make an Image of an Amp"""

    return imageSource.getImage(amp, imageFactory=imageFactory)

def makeImageFromCcd(ccd, imageSource=ButlerImage(), amp=None,
                     isTrimmed=None, correctGain = False, imageFactory=afwImage.ImageU, bin=1,
                     display=False):
    """Make an Image of a Ccd (or just a single amp)
    """

    if isTrimmed is None:
        isTrimmed = ccd.isTrimmed()
    if imageSource:
        imageSource.setTrimmed(isTrimmed)

    if amp:
        ampImage = imageFactory(amp.getAllPixels(isTrimmed).getDimensions())
        ampImage <<= imageSource.getImage(ccd, amp, imageFactory=imageFactory)

        if bin > 1:
            ampImage = afwMath.binImage(ampImage, bin)
            
        return ampImage
    #
    # If the image is raw it may need to be assembled into a full sensor.  The Amp object knows
    # the coordinates system in which the data is being stored on disk.  Since all bounding box
    # information is held in camera coordinates, there is no need to rotate the image after assembly.
    #
    if imageSource:
        if imageSource.isRaw:
            ccdImage = imageFactory(ccd.getAllPixels(isTrimmed))
            for a in ccd:
                im = ccdImage.Factory(ccdImage, a.getAllPixels(isTrimmed), afwImage.LOCAL)
                im <<= imageSource.getImage(ccd, a, imageFactory=imageFactory)
        else:
            ccdImage = imageSource.getImage(ccd, imageFactory=imageFactory)
            if correctGain:
                for a in ccd:
                    samp = ccdImage.Factory(ccdImage, a.getAllPixels(True))
                    samp /= a.getElectronicParams().getGain()

        if bin > 1 and not imageSource.imageIsBinned:
            ccdImage = afwMath.binImage(ccdImage, bin)
        if display:
            showCcd(ccd, ccdImage=ccdImage, isTrimmed=isTrimmed)
    else:
        dims = ccd.getAllPixels(isTrimmed).getDimensions()
        ccdImage = imageFactory(dims[0]//bin, dims[1]//bin)
            
    return ccdImage

def trimExposure(ccdImage, ccd=None, subtractBias=False, rotate=False, modifyRawData=False):
    """Trim a raw CCD Exposure, returning a new (trimmed, possibly rotated) Exposure

If ccd isn't provided it'll be found in the input exposure, ccdImage
If subtractBias is true, subtract the bias levels;  if modifyRawData is true, also subtract it from
the input exposure.
    """

    if not ccd:
        ccd = cameraGeom.cast_Ccd(ccdImage.getDetector())
    
    if rotate:
        nQuarter = ccd.getOrientation().getNQuarter()
        if nQuarter != 0:
            ccdImage.setMaskedImage(afwMath.rotateImageBy90(ccdImage.getMaskedImage(), nQuarter))

    dim = ccd.getAllPixels(True).getDimensions()
    trimmedImage = ccdImage.Factory(dim)

    for a in ccd:
        data = ccdImage.Factory(ccdImage, a.getDataSec(False), afwImage.LOCAL).getMaskedImage()
        if subtractBias:
            bias = ccdImage.getMaskedImage()[a.getBiasSec(False)]
            biasVal = afwMath.makeStatistics(bias, afwMath.MEANCLIP).getValue()
            biasVal = int(biasVal + 0.5) if isinstance(bias, afwImage.MaskedImageU) else biasVal
            data -= biasVal
            if modifyRawData:           # Subtract from input image too
                ccdImage.getMaskedImage()[a.getAllPixels(False)] -= biasVal                    

        tdata = trimmedImage.Factory(trimmedImage, a.getDataSec(True), afwImage.LOCAL).getMaskedImage()
        tdata <<= data

    ccd.setTrimmed(True)
    trimmedImage.setDetector(ccd)

    return trimmedImage

def showCcd(ccd, ccdImage="", amp=None, ccdOrigin=None, isTrimmed=None, frame=None, overlay=True, bin=1):
    """Show a CCD on ds9.  If cameraImage is "", an image will be created based on the properties
of the detectors"""

    if ccdOrigin is None:
        ccdOrigin = afwGeom.PointD(0, 0)
    else:
        ccdOrigin = ccd.getPositionFromPixel(afwGeom.PointD(0, 0)).getMm() # + afwGeom.ExtentD(ccdOrigin)

    if isTrimmed is None:
        isTrimmed = ccd.isTrimmed()

    if ccdImage == "":
        ccdImage = makeImageFromCcd(ccd, bin=bin)

    if ccdImage:
        title = ccd.getId().getName()
        if amp:
            title += ":%d" % amp.getId().getSerial()
        if isTrimmed:
            title += "(trimmed)"
        ds9.mtv(ccdImage, frame=frame, title=title)

    if not overlay:
        return

    with ds9.Buffering():
        if amp:
            bboxes = [(amp.getAllPixels(isTrimmed), 0.49, None),]
            xy0 = bboxes[0][0].getMin()
            if not isTrimmed:
                bboxes.append((amp.getBiasSec(), 0.49, ds9.RED)) 
                bboxes.append((amp.getDataSec(), 0.49, ds9.BLUE))

            for bbox, borderWidth, ctype in bboxes:
                bbox = bbox.clone()
                bbox.shift(-afwGeom.ExtentI(xy0))
                displayUtils.drawBBox(bbox, borderWidth=borderWidth, ctype=ctype, frame=frame, bin=bin)

            return

        nQuarter = ccd.getOrientation().getNQuarter()
    #    ccdDim = cameraGeom.rotateBBoxBy90(ccd.getAllPixels(isTrimmed), nQuarter,
    #           ccd.getAllPixels(isTrimmed).getDimensions()).getDimensions()
        for a in ccd:
            bbox = a.getAllPixels(isTrimmed)
    #        if nQuarter != 0:
    #            bbox = cameraGeom.rotateBBoxBy90(bbox, nQuarter, ccdDim)

            displayUtils.drawBBox(bbox, origin=ccdOrigin, borderWidth=0.49,
                                  frame=frame, bin=bin)

            if not isTrimmed:
                for bbox, ctype in ((a.getBiasSec(), ds9.RED), (a.getDataSec(), ds9.BLUE)):
    #                if nQuarter != 0:
    #                    bbox = cameraGeom.rotateBBoxBy90(bbox, nQuarter, ccdDim)
                    displayUtils.drawBBox(bbox, origin=ccdOrigin,
                                          borderWidth=0.49, ctype=ctype, frame=frame, bin=bin)
            # Label each Amp
            ap = a.getAllPixels(isTrimmed)
            xc, yc = (ap.getMin()[0] + ap.getMax()[0])//2, (ap.getMin()[1] +
                    ap.getMax()[1])//2
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
            ccdHeight = ccd.getAllPixels(isTrimmed).getHeight()
            ccdWidth = ccd.getAllPixels(isTrimmed).getWidth()
            xc -= 0.5*ccdHeight
            yc -= 0.5*ccdWidth

            xc, yc = 0.5*ccdHeight + c*xc + s*yc, 0.5*ccdWidth + -s*xc + c*yc

            if ccdOrigin:
                xc += ccdOrigin[0]
                yc += ccdOrigin[1]

            ds9.dot(str(ccd.findAmp(cen).getId().getSerial()), xc/bin, yc/bin, frame=frame)

        displayUtils.drawBBox(ccd.getAllPixels(isTrimmed), origin=ccdOrigin,
                              borderWidth=0.49, ctype=ds9.MAGENTA, frame=frame, bin=bin)

def makeImageFromRaft(raft, imageSource=ButlerImage(), raftCenter=None,
                      imageFactory=afwImage.ImageU, bin=1):
    """Make an Image of a Raft"""

    if raftCenter is None:
        raftCenter = afwGeom.Point2I(raft.getAllPixels().getDimensions()[0]//2,
                raft.getAllPixels().getDimensions()[1]//2)

    dimensions = afwGeom.Extent2I(raft.getAllPixels().getDimensions()[0]//bin,
            raft.getAllPixels().getDimensions()[1]//bin)
    raftImage = imageFactory(dimensions)

    for det in raft:
        ccd = cameraGeom.cast_Ccd(det)
        
        bbox = ccd.getAllPixels(True)
        cen = ccd.getCenterPixel()
        origin = afwGeom.Point2I(cen)
        origin -= bbox.getDimensions()/2
        origin += afwGeom.Extent2I(raftCenter)
        dims = afwGeom.Extent2I(bbox.getDimensions()[0]//bin,
                bbox.getDimensions()[1]//bin)
        bbox = afwGeom.Box2I(afwGeom.Point2I((origin.getX() + bbox.getMinX())//bin,
                                             (origin.getY() + bbox.getMinY())//bin),
                             dims)

        ccdImage = raftImage.Factory(raftImage, bbox, afwImage.LOCAL)

        dataImage = makeImageFromCcd(ccd, imageSource, imageFactory=imageFactory, isTrimmed=True, bin=bin)
        
        if ccdImage.getDimensions() == dataImage.getDimensions():
            ccdImage <<= dataImage
        else:
            delta = ccdImage.getDimensions() - dataImage.getDimensions()

            if imageSource.gravity == "N":
                x0, y0 = delta[0]//2, delta[1]
            elif imageSource.gravity == "S":
                x0, y0 = delta[0]//2, 0
            elif imageSource.gravity == "E":
                x0, y0 = delta[0], delta[1]//2
            elif imageSource.gravity == "W":
                x0, y0 = 0, delta[1]//2
            else:
                x0, y0 = delta[0]//2, delta[1]//2
            subCcdImage = ccdImage.Factory(ccdImage, afwGeom.BoxI(afwGeom.PointI(x0, y0), dataImage.getDimensions()))
            subCcdImage <<= dataImage
            del subCcdImage

    return raftImage

def showRaft(raft, imageSource=ButlerImage(), raftOrigin=None, frame=None, overlay=True, bin=1):
    """Show a Raft on ds9.

If imageSource isn't None, create an image using the images specified by imageSource"""

    if raftOrigin is None:
        raftOrigin = afwGeom.PointD(0, 0)
    else:
        raftOrigin = raft.getPositionFromPixel(afwGeom.PointD(0, 0)).getMm()

    raftCenter = afwGeom.Point2I(raft.getAllPixels().getDimensions()/2)
    raftCenter += afwGeom.ExtentI(int(raftOrigin[0]), int(raftOrigin[1]))

    if imageSource is None:
        raftImage = None
    elif isinstance(imageSource, GetCcdImage):
        raftImage = makeImageFromRaft(raft, imageSource=imageSource, raftCenter=raftCenter, bin=bin)
    else:
        raftImage = imageSource

    if raftImage:
        ds9.mtv(raftImage, frame=frame, title=raft.getId().getName())

    if not raftImage and not overlay:
        return

    with ds9.Buffering():
        for det in raft:
            ccd = cameraGeom.cast_Ccd(det)

            bbox = ccd.getAllPixels(True)
            origin = ccd.getPositionFromPixel(afwGeom.PointD(0, 0)).getMm()
            #origin -= afwGeom.ExtentD(raftCenter)

            if True:
                name = ccd.getId().getName()
            else:
                name = str(ccd.getCenter())

            ds9.dot(name, (origin[0] + 0.5*bbox.getWidth())/bin,
                          (origin[1] + 0.4*bbox.getHeight())/bin, frame=frame)

            showCcd(ccd, None, isTrimmed=True, frame=frame, ccdOrigin=origin, overlay=overlay, bin=bin)

def makeImageFromCamera(camera, imageSource=None, imageFactory=afwImage.ImageU, bin=1):
    """Make an Image of a Camera"""

    cameraImage = imageFactory(camera.getAllPixels().getDimensions()/bin)
    for det in camera:
        raft = cameraGeom.cast_Raft(det);
        bbox = raft.getAllPixels()
        origin = camera.getCenterPixel() + afwGeom.Extent2D(raft.getCenterPixel()) - \
                 afwGeom.Extent2D(bbox.getWidth()/2, bbox.getHeight()/2)               
        dimensions = afwGeom.Extent2I(bbox.getDimensions()[0]//bin,
                bbox.getDimensions()[1]//bin)
        bbox = afwGeom.Box2I(afwGeom.Point2I(int((origin.getX() + bbox.getMinX())//bin),
                                           int((origin.getY() + bbox.getMinY())//bin)),
                             dimensions)

        im = cameraImage.Factory(cameraImage, bbox, afwImage.LOCAL)

        im <<= makeImageFromRaft(raft, imageSource,
                                 raftCenter=None,
                                 imageFactory=imageFactory, bin=bin)
        serial = raft.getId().getSerial()
        im += serial if serial > 0 else 0

    return cameraImage

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class PyCcd(object):
    """A picklable partial implementation of cameraGeom.Ccd"""
    def __init__(self, ccd):
        self._id = PyCcdId(ccd.getId().getSerial(), ccd.getId().getName())
        self._allPixels = ccd.getAllPixels()
        self._orientation = PyCcdOrientation(ccd.getOrientation())

    def getId(self):
        return self._id

    def getAllPixels(self, isTrimmed=False):
        return self._allPixels

    def getOrientation(self):
        return self._orientation        

class PyCcdId(object):
    """A picklable partial implementation of cameraGeom.Id"""
    def __init__(self, serial, name):
        self.serial = serial
        self.name = name

    def getName(self):
        return self.name

    def getSerial(self):
        return self.serial

class PyCcdOrientation(object):
    """A picklable partial implementation of cameraGeom.Orientation"""
    def __init__(self, orientation):
        self._nQuarter = orientation.getNQuarter()

    def getNQuarter(self):
        return self._nQuarter

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class MakeImageFromCcdWorker(object):
    def __init__(self, verbose=False, logLevel=None):
        self.verbose = verbose
        self.logLevel = logLevel

    def __call__(self, payload):
        serialNo, args, kwargs = payload

        if self.logLevel is not None:
            pexLog.getDefaultLog().setThresholdFor("CameraMapper", self.logLevel)
           
        try:
            img = makeImageFromCcd(*args, **kwargs)
        except Exception, e:
            if self.verbose:
                print >> sys.stderr, e
                img = None

        if self.verbose > 1:
            print "Returning image for CCD%03d [PID %d]" % (serialNo, os.getpid())

        return serialNo, img

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def showCamera(camera, imageSource=ButlerImage(), imageFactory=afwImage.ImageF, nJob=None,
               onlySerials=None, bin=1, border=5, frame=None, overlay=True, title="", ctype=ds9.GREEN, names=False):
    """Show a Camera on ds9 (with the specified frame); if overlay show the IDs and detector boundaries

    If onlySerials is provided, only show CCDs with those serial numbers

If imageSource is provided its getImage method will be called to return a CCD image (e.g. a
cameraGeom.GetCcdImage object); if it is "", an image will be created based on the properties
of the detectors.  If it's None then no image is created or displayed (useful if overlay is True)
"""
    #
    # Figure out how big an image I need, and where the CCDs fit
    #
    cameraBbox = afwGeom.BoxI()
    ccdBboxes = {}
    for raft in camera:
        raft = cameraGeom.cast_Raft(raft)
        for ccd in raft:
            if onlySerials and ccd.getId().getSerial() not in onlySerials:
                continue

            ccd = cameraGeom.cast_Ccd(ccd)
            ccd.setTrimmed(True)
            
            width, height = ccd.getAllPixels(True).getDimensions()

            bbox = afwGeom.BoxI()
            for x, y in ((0.0,0.0), (0.0, height - 1), (width - 1, height - 1), (width - 1, 0.0), (0.0, 0.0)):
                position = ccd.getPositionFromPixel(afwGeom.Point2D(x,y)).getPixels(ccd.getPixelSize())
                bbox.include(afwGeom.PointI(int(position.getX()//bin), int(position.getY()//bin)))

            cameraBbox.include(bbox)
            ccdBboxes[ccd.getId().getSerial()] = bbox

    cameraBbox.grow(afwGeom.ExtentI(border, border))

    cameraImage = imageFactory(cameraBbox)
    cameraImage.set(imageSource.background if imageSource else np.nan)

    if cameraImage is not None:         # includes ""
        #
        # If we're using multiprocessing we need to create the jobs now
        #
        makeImageForCcdArgs = []        # list of arguments to be passed to calls to makeImageForCcd
        verbose = True
        if nJob > 1:
            pool = multiprocessing.Pool(nJob)
        #
        # We'll do this in two stages;  first calculate the ccdImage (where the data will go)
        # and then the dataImage (the data), then set one from the other.
        #
        # The split is to allow us to parallelise the calculation of the dataImages
        #
        ccdImages = {}
        for raft in camera:
            raft = cameraGeom.cast_Raft(raft)
            for i, ccd in enumerate(raft):
                ccd = cameraGeom.cast_Ccd(ccd)

                serialNo = ccd.getId().getSerial()
                if not ccdBboxes.has_key(serialNo):
                    continue

                ccdImages[serialNo] = cameraImage.Factory(cameraImage, ccdBboxes[serialNo], afwImage.PARENT)
                #
                # Now figure out how to calculate the dataImage
                #
                args = (PyCcd(ccd),     # we can't pickle a Ccd for multiprocessing
                        imageSource)
                kwargs = dict(imageFactory=imageFactory, isTrimmed=True, bin=bin)

                makeImageForCcdArgs.append((serialNo, args, kwargs))
        #
        # Calculate the dataimages, either in series or parallel
        #
        dataImages = {}
        if nJob <= 1:
            worker = MakeImageFromCcdWorker()
            for args in makeImageForCcdArgs:
                dataImages.update([worker(args)])
        else:
            # We use map_async(...).get(9999) instead of map(...) to workaround a python bug
            # in handling ^C in subprocesses (http://bugs.python.org/issue8296)
            worker = MakeImageFromCcdWorker(verbose, pexLog.Log.FATAL)
            for serialNo, dataImage in pool.map_async(worker, makeImageForCcdArgs).get(9999):

                dataImages[serialNo] = dataImage

            pool.close()
            pool.join()

        for serialNo in ccdImages.keys():
            dataImage, ccdImage = dataImages[serialNo], ccdImages[serialNo]

            if ccdImage.getDimensions() == dataImage.getDimensions():
                ccdImage <<= dataImage
            else:
                delta = ccdImage.getDimensions() - dataImage.getDimensions()

                if imageSource:
                    if imageSource.gravity == "N":
                        x0, y0 = delta[0]//2, delta[1]
                    elif imageSource.gravity == "S":
                        x0, y0 = delta[0]//2, 0
                    elif imageSource.gravity == "E":
                        x0, y0 = delta[0], delta[1]//2
                    elif imageSource.gravity == "W":
                        x0, y0 = 0, delta[1]//2
                    else:
                        x0, y0 = delta[0]//2, delta[1]//2
                else:
                    assert bin > 1
                    x0, y0 = 0, 0       # rounding error in binning

                try:
                    subCcdImage = ccdImage.Factory(ccdImage, afwGeom.BoxI(afwGeom.PointI(x0, y0),
                                                                          dataImage.getDimensions()))
                    subCcdImage <<= dataImage
                    del subCcdImage
                except Exception, e:
                    print "RHL", e
                    import pdb; pdb.set_trace()
                    pass
    #
    # We've got the image, add a WCS
    #
    if frame is not None:
        if imageSource is not None:
            wcs = makeFocalPlaneWcs(camera)
            
            ds9.mtv(cameraImage, title=title, frame=frame, wcs=wcs)

        if overlay:
            xy0 = afwGeom.ExtentI(cameraImage.getXY0())
            
            with ds9.Buffering():
                for raft in camera:
                    raft = cameraGeom.cast_Raft(raft)
                    for ccd in raft:
                        ccd = cameraGeom.cast_Ccd(ccd)

                        serialNo = ccd.getId().getSerial()
                        if not ccdBboxes.has_key(serialNo):
                            continue

                        bbox = ccdBboxes[serialNo]
                        displayUtils.drawBBox(bbox, origin=-xy0, borderWidth=0.49,
                                              ctype=ctype, frame=frame)

                        xy = [0, 0]
                        for i in range(2):
                            xy[i] = 0.5*(bbox.getMin()[i] + bbox.getMax()[i]) - xy0[i]

                        ds9.dot(ccd.getId().getName() if names else str(serialNo), *xy,
                                ctype=ctype, frame=frame)

    return cameraImage

def makeFocalPlaneWcs(camera):
    """Make a WCS for the focal plane geometry (i.e. returning positions in "mm")"""
    import lsst.daf.base as dafBase

    ccd = cameraGeom.cast_Ccd(list(cameraGeom.cast_Raft(list(camera)[0]))[0]) # some random CCD

    md = dafBase.PropertySet()
    pix = afwGeom.PointD(0,0)
    fpPos = ccd.getPositionFromPixel(pix).getMm()

    for i in range(2):
        md.set("CRPOS%d" % i, pix[i])
        md.set("CRVAL%d" % i, fpPos[i])
    md.set("CDELT1", ccd.getPixelSize())

    return afwImage.makeWcs(md)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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

def describeRaft(raft, indent=""):
    """Describe an entire Raft"""
    descrip = []

    size = raft.getSize().getMm()
    descrip.append("%sRaft \"%s\",  %gx%g  BBox %s" % (indent, raft.getId(),
                                                        size[0], size[1], raft.getAllPixels()))
    
    for d in cameraGeom.cast_Raft(raft):
        cenPixel = d.getCenterPixel()
        cen = d.getCenter().getMm()

        descrip.append("%sCcd: %s (%5d, %5d) %s  (%7.1f, %7.1f)" % \
                       ((indent + "    "),
                        d.getAllPixels(True), cenPixel[0], cenPixel[1],
                        cameraGeom.ReadoutCorner(d.getOrientation().getNQuarter()), cen[0], cen[1]))
            
    return "\n".join(descrip)

def describeCamera(camera):
    """Describe an entire Camera"""
    descrip = []

    size = camera.getSize().getMm()
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

    if isinstance(id, int):
        id = cameraGeom.Id(id)

    if isinstance(parent, cameraGeom.Camera):
        for d in parent:
            ccd = findCcd(cameraGeom.cast_Raft(d), id)
            if ccd:
                return ccd
    elif isinstance(parent, cameraGeom.Raft):
        try:
            return cameraGeom.cast_Ccd(parent.findDetector(id))
        except:
            pass
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

def makeDefectsFromFits(filename):
    """Create a dictionary of DefectSets from a fits file with one ccd worth
    of defects per extension.

       The dictionary is indexed by an Id object --- remember to compare by str(id) not object identity
    """
    if not pyfits:
        raise RuntimeError("Import of pyfits failed, so makeDefectsFromFits isn't available")

    hdulist = pyfits.open(filename)
    defects = {}
    for hdu in hdulist[1:]:
        id = cameraGeom.Id(hdu.header['serial'], hdu.header['name'])
        data = hdu.data
        defectList = []
        for i in range(len(data)):
            bbox = afwGeom.Box2I(
                        afwGeom.Point2I(int(data[i]['x0']), int(data[i]['y0'])),\
		        afwGeom.Extent2I(int(data[i]['width']), int(data[i]['height'])))
            defectList.append(afwImage.DefectBase(bbox))
        defects[id] = defectList
    return defects

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def makeDefects(geomPolicy):
    """Create a dictionary of DefectSets from a pexPolicy::Policy

The dictionary is indexed by an Id object --- remember to compare by str(id) not object identity
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

                bbox = afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Point2I(x1, y1))
                defects.push_back(afwImage.DefectBase(bbox))

    return defectsDict



def makeDefaultCcd(box, **kwargs):
    """
    Make a Ccd object for a Box2I object suitable to be used as Detector in an Exposure
    **kwargs may specify params needed to construct ElectronicParams, Amp, and Ccd.
    Defaults are:
       ElectronicParams: gain (1.0), rdnoise (5.0), saturation (60000)
       Amp:              detId (1), biasSec (empty Box2I), dataSec (Box2I(img.xy0, img.getDimentions()))
       Ccd:              pixelSize (1.0)
    This is only intented to be a useful factory for Detectors needed to test simple images.
    """
    
    # build the Electronics
    gain          = kwargs.get("gain", 1.0)
    rdnoise       = kwargs.get("rdnoise", 5.0)
    saturation    = kwargs.get("saturation", 60000)
    elec = cameraGeom.ElectronicParams(gain, rdnoise, saturation)

    # build the Amp
    detId         = kwargs.get("detId", cameraGeom.Id(1))
    allPixels     = box
    biasSec       = kwargs.get("biasSec", afwGeom.Box2I())
    dataSec       = kwargs.get("dataSec", box)
    amp = cameraGeom.Amp(detId, allPixels, biasSec, dataSec, elec)

    # build the Ccd
    pixelSize     = kwargs.get("pixelSize", 1.0)
    ccd = cameraGeom.Ccd(detId, pixelSize)
    ccd.addAmp(amp)

    return ccd

            

def makeCcdFromImage(img, **kwargs):
    """
    Make a Ccd object for an afw::Image object suitable to be used as Detector in an Exposure
    **kwargs may specify params needed to construct Electronics, Amp, and Ccd.  Defaults are:
       Electronics: gain (1.0), rdnoise (5.0), saturation (60000)
       Amp:         detId (1), biasSec (empty Box2I), dataSec (Box2I(img.xy0, img.getDimentions()))
       Ccd:         pixelSize (1.0)
    This is only intented to be a useful factory for Detectors needed to test simple images.
    """
    
    allPixels     = afwGeom.Box2I(img.getXY0(), img.getDimensions())

    return makeDefaultCcd(allPixels, **kwargs)
    
