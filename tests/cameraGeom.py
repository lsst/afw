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

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept
import lsst.pex.policy as pexPolicy
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
#import lsst.afw.math.mathLib as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

import lsst.afw.cameraGeom as cameraGeom

try:
    type(display)
except NameError:
    display = False

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
        ccdId = cameraGeom.Id(ccdPol.get("serial"), ccdPol.get("name"))

    ccd = cameraGeom.Ccd(ccdId, pixelSize)

    if nCol*nRow != len(ccdPol.getArray("Amp")):
        raise RuntimeError, ("Expected location of %d amplifiers, got %d" % \
                             (nCol*nRow, len(ccdPol.getArray("Amp"))))

    if ccdInfo is None:
        ampSerial = [0]
    else:
        ampSerial = ccdInfo.get("ampSerial", [0])
        ampSerial0 = ampSerial[0]           # used in testing
        
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

        if c == "LLC":
            c = cameraGeom.Amp.LLC
            biasSec = afwImage.BBox(afwImage.PointI(extended + width, preRows), overclockH, height)
            dataSec = afwImage.BBox(afwImage.PointI(extended, preRows), width, height)
        elif c == "LRC":
            c = cameraGeom.Amp.LRC
            biasSec = afwImage.BBox(afwImage.PointI(0, preRows), overclockH, height)
            dataSec = afwImage.BBox(afwImage.PointI(overclockH, preRows), width, height)
        else:
            raise RuntimeError, ("Unknown readoutCorner %s" % c)

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
        raftId = cameraGeom.Id(raftPol.get("serial"), raftPol.get("name"))

    raft = cameraGeom.Raft(raftId)

    if nCol*nRow != len(raftPol.getArray("Ccd")):
        raise RuntimeError, ("Expected location of %d amplifiers, got %d" % \
                             (nCol*nRow, len(raftPol.getArray("Ccd"))))

    for ccdPol in raftPol.getArray("Ccd"):
        Col, Row = ccdPol.getArray("index")
        xc, yc = ccdPol.getArray("offset")
        pitch, roll, yaw = [float(math.radians(a)) for a in ccdPol.getArray("orientation")]

        if Col not in range(nCol) or Row not in range(nRow):
            raise RuntimeError, ("Amp location %d, %d is not in 0..%d, 0..%d" % (Col, Row, nCol, nRow))

        ccdId = cameraGeom.Id(ccdPol.get("serial"), ccdPol.get("name"))
        ccd = makeCcd(geomPolicy, ccdId, ccdInfo=ccdInfo)

        raft.addDetector(afwGeom.Point2I.makeXY(Col, Row),
                         afwGeom.Point2D.makeXY(xc, yc), cameraGeom.Orientation(pitch, roll, yaw), ccd)

    if raftInfo is not None:
        raftInfo.clear()
        raftInfo["ampSerial"] = ccdInfo["ampSerial"]
        raftInfo["name"] = raft.getId().getName()
        raftInfo["pixelSize"] = ccd.getPixelSize()
        raftInfo["width"] =  nCol*ccd.getAllPixels(True).getWidth()
        raftInfo["height"] = nRow*ccd.getAllPixels(True).getHeight()

    return raft

def makeCamera(geomPolicy, cameraId=None, cameraInfo={}):
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
    camera = cameraGeom.Camera(cameraId)

    for raftPol in cameraPol.getArray("Raft"):
        Col, Row = raftPol.getArray("index")
        xc, yc = raftPol.getArray("offset")
        pitch, roll, yaw = [float(math.radians(a)) for a in raftPol.getArray("orientation")]

        raftId = cameraGeom.Id(raftPol.get("serial"), raftPol.get("name"))
        raft = makeRaft(geomPolicy, raftId, raftInfo)
        camera.addDetector(afwGeom.Point2I.makeXY(Col, Row),
                           afwGeom.Point2D.makeXY(xc, yc), cameraGeom.Orientation(pitch, roll, yaw), raft)

    cameraInfo.clear()
    cameraInfo["ampSerial"] = raftInfo["ampSerial"]
    cameraInfo["name"] = camera.getId().getName()
    cameraInfo["width"] =  nCol*raft.getAllPixels().getWidth()
    cameraInfo["height"] = nRow*raft.getAllPixels().getHeight()
    cameraInfo["pixelSize"] = raft.getPixelSize()

    return camera

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def makeImageFromCcd(ccd):
    """Make an Image of a Ccd"""

    ccdImage = afwImage.ImageU(ccd.getAllPixels().getDimensions())
    for a in ccd:
        im = ccdImage.Factory(ccdImage, a.getAllPixels())
        im += int(a.getElectronicParams().getReadNoise())
        im = ccdImage.Factory(ccdImage, a.getDataSec())
        im += int(1 + 100*a.getElectronicParams().getGain() + 0.5)

    return ccdImage

def showCcd(ccd, ccdImage="", ccdOrigin=None, isTrimmed=None, frame=None):
    """Show a CCD on ds9.  If cameraImage isn't "", an image will be created based on the properties
of the detectors"""
    
    if isTrimmed is None:
        isTrimmed = ccd.isTrimmed()

    if ccdImage == "":
        ccdImage = makeImageFromCcd(ccd)

    if ccdImage is not None:
        title = ccd.getId().getName()
        if isTrimmed:
            title += "(trimmed)"
        ds9.mtv(ccdImage, frame=frame, title=title)

    for a in ccd:
        displayUtils.drawBBox(a.getAllPixels(isTrimmed), origin=ccdOrigin, borderWidth=0, frame=frame)
        if not isTrimmed:
            displayUtils.drawBBox(a.getBiasSec(), origin=ccdOrigin, ctype=ds9.RED, frame=frame)
            displayUtils.drawBBox(a.getDataSec(), origin=ccdOrigin, ctype=ds9.BLUE, frame=frame)
        # Label each Amp
        ap = a.getAllPixels(isTrimmed)
        xc, yc = (ap.getX0() + ap.getX1())//2, (ap.getY0() + ap.getY1())//2
        cen = afwGeom.Point2I.makeXY(xc, yc)
        if ccdOrigin:
            xc += ccdOrigin[0]
            yc += ccdOrigin[1]
        ds9.dot(str(ccd.findAmp(cen).getId().getSerial()), xc, yc, frame=frame)

    displayUtils.drawBBox(ccd.getAllPixels(isTrimmed), origin=ccdOrigin,
                          borderWidth=0.5, ctype=ds9.MAGENTA, frame=frame)

def makeImageFromRaft(raft):
    """Make an Image of a Raft"""

    raftImage = afwImage.ImageU(raft.getAllPixels().getDimensions())
    for dl in raft:
        det = dl.getDetector();
        bbox = det.getAllPixels(True).clone()
        bbox.shift(dl.getOrigin()[0], dl.getOrigin()[1])
        im = raftImage.Factory(raftImage, bbox)
        im.set(det.getId().getSerial())

    return raftImage

def showRaft(raft, raftImage="", raftOrigin=None, frame=None):
    """Show a Raft on ds9.  If cameraImage isn't "", an image will be created based on the
properties of the detectors"""
    if raftImage == "":
        raftImage = makeImageFromRaft(raft)

    if raftImage is not None:
        ds9.mtv(raftImage, frame=frame, title=raft.getId().getName())

    for dl in raft:
        ccd = cameraGeom.cast_Ccd(dl.getDetector())
        
        bbox = ccd.getAllPixels(True)
        origin = dl.getOrigin()
        if raftOrigin:
            origin += afwGeom.Extent2I(raftOrigin)
            
        ds9.dot(ccd.getId().getName(),
                origin[0] + bbox.getWidth()/2, origin[1] + bbox.getHeight()/2, frame=frame)

        showCcd(ccd, None, isTrimmed=True, frame=frame, ccdOrigin=origin)

def makeImageFromCamera(camera):
    """Make an Image of a Camera"""

    cameraImage = afwImage.ImageU(camera.getAllPixels().getDimensions())
    for dl in camera:
        raft = dl.getDetector();
        bbox = raft.getAllPixels().clone()
        bbox.shift(dl.getOrigin()[0], dl.getOrigin()[1])
        im = cameraImage.Factory(cameraImage, bbox)
        im.set(raft.getId().getSerial())

    return cameraImage

def showCamera(camera, cameraImage="", frame=None):
    """Show a Camera on ds9.  If cameraImage isn't "", an image will be created based
on the properties of the detectors"""
    
    if cameraImage == "":
        cameraImage = makeImageFromCamera(camera)

    if cameraImage is not None:
        ds9.mtv(cameraImage, frame=frame, title=camera.getId().getName())

    for dl in camera:
        raft = cameraGeom.cast_Raft(dl.getDetector())
        
        bbox = raft.getAllPixels()
        ds9.dot(raft.getId().getName(),
                dl.getOrigin()[0] + bbox.getWidth()/2, dl.getOrigin()[1] + bbox.getHeight()/2, frame=frame)

        showRaft(raft, None, frame=frame, raftOrigin=dl.getOrigin())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def trimCcd(ccd, ccdImage=""):
    if ccdImage == "":
        ccdImage = makeImageFromCcd(ccd)

    if ccd.isTrimmed():
        return ccdImage

    ccd.setTrimmed(True)

    if ccdImage is not None:
        trimmedImage = ccdImage.Factory(ccd.getAllPixels().getDimensions())
        for a in ccd:
            data =      ccdImage.Factory(ccdImage, a.getDataSec(False))
            tdata = trimmedImage.Factory(trimmedImage, a.getDataSec())
            tdata <<= data
    else:
        trimmedImage = None

    return trimmedImage

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class CameraGeomTestCase(unittest.TestCase):
    """A test case for camera geometry"""

    def assertEqualPoint(self, lhs, rhs):
        """Workaround the Point2D logical operator returning a Point not a bool"""
        if (lhs != rhs)[0] or (lhs != rhs)[1]:
            self.assertTrue(False, "%s != %s" % (lhs, rhs))

    def setUp(self):
        CameraGeomTestCase.ampSerial = [0] # an array so we pass the value by reference

        policyFile = pexPolicy.DefaultPolicyFile("afw", "CameraGeomDictionary.paf", "policy")
        defPolicy = pexPolicy.Policy.createPolicy(policyFile, policyFile.getRepositoryPath(), True)

        polFile = pexPolicy.DefaultPolicyFile("afw", "TestCameraGeom.paf", "tests")
        self.geomPolicy = pexPolicy.Policy.createPolicy(polFile)
        self.geomPolicy.mergeDefaults(defPolicy.getDictionary())
    
    def tearDown(self):
        pass

    def testDictionary(self):
        """Test the camera geometry dictionary"""

        if False:
            for r in self.geomPolicy.getArray("Raft"):
                print "raft", r
            for c in self.geomPolicy.getArray("Ccd"):
                print "ccd", c
            for a in self.geomPolicy.getArray("Amp"):
                print "amp", a

    def testId(self):
        """Test cameraGeom.Id"""
        try:
            cameraGeom.Id(-1)
        except pexExcept.LsstCppException:
            pass
        else:
            self.assertFalse("Should raise an exception on -ve serials")

        self.assertTrue(cameraGeom.Id(1) == cameraGeom.Id(1))
        self.assertFalse(cameraGeom.Id(1) == cameraGeom.Id(100))
        
        self.assertTrue(cameraGeom.Id("AA") == cameraGeom.Id("AA"))
        self.assertFalse(cameraGeom.Id("AA") == cameraGeom.Id("BB"))
        
        self.assertTrue(cameraGeom.Id(1, "AA") == cameraGeom.Id(1, "AA"))
        self.assertFalse(cameraGeom.Id(1, "AA") == cameraGeom.Id(2, "AA"))
        self.assertFalse(cameraGeom.Id(1, "AA") == cameraGeom.Id(1, "BB"))

    def testCcd(self):
        """Test if we can build a Ccd out of Amps"""

        #print >> sys.stderr, "Skipping testCcd"; return

        ccdId = cameraGeom.Id("CCD")
        ccdInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        ccd = makeCcd(self.geomPolicy, ccdId, ccdInfo=ccdInfo)
        if display:
            showCcd(ccd, frame=0)
        else:
            ccdImage = None

        for i in range(2):
            self.assertEqual(ccd.getSize()[i], ccdInfo["pixelSize"]*ccd.getAllPixels(True).getDimensions()[i])

        self.assertEqual(ccd.getId().getName(), ccdInfo["name"])
        self.assertEqual(ccd.getAllPixels().getWidth(), ccdInfo["width"])
        self.assertEqual(ccd.getAllPixels().getHeight(), ccdInfo["height"])
        self.assertEqual([a.getId().getSerial() for a in ccd],
                         range(ccdInfo["ampIdMin"], ccdInfo["ampIdMax"] + 1))

        id = cameraGeom.Id("ID%d" % ccdInfo["ampIdMax"])
        self.assertTrue(ccd.findAmp(id), id)

        self.assertEqual(ccd.findAmp(afwGeom.Point2I.makeXY(10, 10)).getId().getSerial(), ccdInfo["ampIdMin"])

        self.assertEqual(ccd.getAllPixels().getLLC(),
                         ccd.findAmp(afwGeom.Point2I.makeXY(10, 10)).getAllPixels().getLLC())

        self.assertEqual(ccd.getAllPixels().getURC(),
                         ccd.findAmp(afwGeom.Point2I.makeXY(ccdInfo["width"] - 1,
                                                            ccdInfo["height"] - 1)).getAllPixels().getURC())
        #
        # Test mapping pixel <--> mm
        #
        pix = afwGeom.Point2I.makeXY(100, 200)
        pos = afwGeom.Point2D.makeXY(1.0, 2.0)
        #
        # Map pix into untrimmed coordinates
        #
        amp = ccd.findAmp(pix)
        corr = amp.getDataSec(False).getLLC() - amp.getDataSec(True).getLLC()
        pix += afwGeom.Extent2I(afwGeom.Point2I.makeXY(corr[0], corr[1]))
        
        self.assertEqual(ccd.getIndexFromPosition(pos), pix)

        self.assertEqualPoint(ccd.getPositionFromIndex(pix), pos)
        #
        # Trim the CCD and try again
        #
        trimmedImage = trimCcd(ccd)

        if display:
            showCcd(ccd, trimmedImage, frame=1)

        a = ccd.findAmp(cameraGeom.Id("ID%d" % ccdInfo["ampIdMin"]))
        self.assertEqual(a.getDataSec(), afwImage.BBox(afwImage.PointI(0, 0),
                                                       ccdInfo["ampWidth"], ccdInfo["ampHeight"]))

        self.assertEqual(ccd.getSize()[0], ccdInfo["pixelSize"]*ccdInfo["trimmedWidth"])
        self.assertEqual(ccd.getSize()[1], ccdInfo["pixelSize"]*ccdInfo["trimmedHeight"])
        #
        # Test mapping pixel <--> mm
        #
        pix = afwGeom.Point2I.makeXY(100, 200)
        pos = afwGeom.Point2D.makeXY(1.0, 2.0)
        
        self.assertEqualPoint(ccd.getIndexFromPosition(pos), pix)
        self.assertEqualPoint(ccd.getPositionFromIndex(pix), pos)

    def testRaft(self):
        """Test if we can build a Raft out of Ccds"""

        #print >> sys.stderr, "Skipping testRaft"; return

        raftId = cameraGeom.Id("Raft")
        raftInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        raft = makeRaft(self.geomPolicy, raftId, raftInfo=raftInfo)

        if display:
            showRaft(raft, frame=2)

        if False:
            print "Raft Name \"%s\", serial %d,  BBox %s" % \
                  (raft.getId().getName(), raft.getId().getSerial(), raft.getAllPixels())

            for d in raft:
                print d.getOrigin(), d.getDetector().getAllPixels(True)

            print "Size =", raft.getSize()

        self.assertEqual(raft.getAllPixels().getWidth(), raftInfo["width"])
        self.assertEqual(raft.getAllPixels().getHeight(), raftInfo["height"])

        for x, y, serial in [(0, 0, 7), (150, 250, 23), (250, 250, 31)]:
            det = raft.findDetector(afwGeom.Point2I.makeXY(x, y)).getDetector()
            ccd = cameraGeom.cast_Ccd(det)
            if False:
                print x, y, det.getId().getName(), \
                      ccd.findAmp(afwGeom.Point2I.makeXY(150, 152), True).getId().getSerial()
            self.assertEqual(ccd.findAmp(afwGeom.Point2I.makeXY(150, 152), True).getId().getSerial(), serial)

        name = "C:0,2"
        self.assertEqual(raft.findDetector(cameraGeom.Id(name)).getDetector().getId().getName(), name)
        #
        # This test isn't really right as we don't allow for e.g. gaps between Detectors.  Well, the
        # test is right but getSize() isn't
        #
        for i in range(2):
            self.assertEqual(raft.getSize()[i], raftInfo["pixelSize"]*raft.getAllPixels().getDimensions()[i])
        
    def testCamera(self):
        """Test if we can build a Camera out of Rafts"""

        #print >> sys.stderr, "Skipping testCamera"; return

        cameraInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        camera = makeCamera(self.geomPolicy, cameraInfo=cameraInfo)

        if display:
            showCamera(camera, frame=3)

        if not False:
            print "Camera Name \"%s\", serial %d,  BBox %s" % \
                  (camera.getId().getName(), camera.getId().getSerial(), camera.getAllPixels())

            for d in camera:
                print "Raft:", d.getOrigin(), d.getDetector().getAllPixels()

            print "Camera size =", camera.getSize()

        self.assertEqual(camera.getAllPixels().getWidth(), cameraInfo["width"])
        self.assertEqual(camera.getAllPixels().getHeight(), cameraInfo["height"])

        for rx, ry, cx, cy, serial in [(0, 0,     0, 0, 7),  (0,   0,   150, 250, 23),
                                       (600, 300, 0, 0, 55), (600, 300, 150, 250, 71)]:
            raft = cameraGeom.cast_Raft(camera.findDetector(afwGeom.Point2I.makeXY(rx, ry)).getDetector())

            ccd = cameraGeom.cast_Ccd(raft.findDetector(afwGeom.Point2I.makeXY(cx, cy)).getDetector())
            if False:
                print rx, ry, cx, cy, raft.getId().getName(), \
                      ccd.findAmp(afwGeom.Point2I.makeXY(150, 152), True).getId().getSerial()
            self.assertEqual(ccd.findAmp(afwGeom.Point2I.makeXY(150, 152), True).getId().getSerial(), serial)

        name = "R:1,0"
        self.assertEqual(camera.findDetector(cameraGeom.Id(name)).getDetector().getId().getName(), name)
        #
        # This test isn't really right as we don't allow for e.g. gaps between Rafts.  Well, the
        # test is right but getSize() isn't
        #
        for i in range(2):
            self.assertEqual(camera.getSize()[i],
                             cameraInfo["pixelSize"]*camera.getAllPixels().getDimensions()[i])

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(CameraGeomTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
