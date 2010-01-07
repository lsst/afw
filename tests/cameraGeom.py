#!/usr/bin/env python
"""
Tests for SpatialCell

Run with:
   python SpatialCell.py
or
   python
   >>> import SpatialCell; SpatialCell.run()
"""

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

def showCcd(ccd, ccdImage, ccdOrigin=None, frame=None):
    if ccdImage:
        title = ccd.getId().getName()
        if ccd.isTrimmed():
            title += "(trimmed)"
        ds9.mtv(ccdImage, frame=frame, title=title)

    for a in ccd:
        if a.getBiasSec().getWidth():
            displayUtils.drawBBox(a.getBiasSec(), origin=ccdOrigin, ctype=ds9.RED, frame=frame)
        displayUtils.drawBBox(a.getDataSec(), origin=ccdOrigin, ctype=ds9.BLUE, frame=frame)
        displayUtils.drawBBox(a.getAllPixels(), origin=ccdOrigin, borderWidth=0.25, frame=frame)
        # Label each Amp
        ap = a.getAllPixels()
        xc, yc = (ap.getX0() + ap.getX1())//2, (ap.getY0() + ap.getY1())//2
        cen = afwGeom.Point2I.makeXY(xc, yc)
        if ccdOrigin:
            xc += ccdOrigin[0]
            yc += ccdOrigin[1]
        ds9.dot(str(ccd.findAmp(cen).getId().getSerial()), xc, yc, frame=frame)

    displayUtils.drawBBox(ccd.getAllPixels(), borderWidth=0.49, ctype=ds9.MAGENTA, frame=frame)

def showRaft(raft, raftImage, raftOrigin=None, frame=None):
    if raftImage:
        ds9.mtv(raftImage, frame=frame, title=raft.getId().getName())

    for dl in raft:
        ccd = cameraGeom.cast_Ccd(dl.getDetector())
        ccd.setTrimmed(True)
        
        bbox = ccd.getAllPixels(True)
        origin = dl.getOrigin()
        if raftOrigin:
            origin += afwGeom.Extent2I(raftOrigin)
            
        ds9.dot(ccd.getId().getName(),
                origin[0] + bbox.getWidth()/2, origin[1] + bbox.getHeight()/2, frame=frame)

        showCcd(ccd, None, frame=frame, ccdOrigin=origin)

def showCamera(camera, cameraImage, frame=None):
    ds9.mtv(cameraImage, frame=frame, title=camera.getId().getName())

    for dl in camera:
        raft = cameraGeom.cast_Raft(dl.getDetector())
        
        bbox = raft.getAllPixels(True)
        ds9.dot(raft.getId().getName(),
                dl.getOrigin()[0] + bbox.getWidth()/2, dl.getOrigin()[1] + bbox.getHeight()/2, frame=frame)

        showRaft(raft, None, frame=frame, raftOrigin=dl.getOrigin())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def trimCcd(ccd, ccdImage):
    ccd.setTrimmed(True)

    if ccdImage:
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

    ampSerial = ccdSerial = raftSerial = 0

    def assertEqualPoint(self, lhs, rhs):
        """Workaround the Point2D logical operator returning a Point not a bool"""
        if (lhs != rhs)[0] or (lhs != rhs)[1]:
            self.assertTrue(False, "%s != %s" % (lhs, rhs))

    def makeCcd(self, geomPolicy, ccdId=None, makeImage=False):
        """Build a Ccd from a set of amplifiers given a suitable pex::Policy"""
        ccdPol = geomPolicy.get("Ccd")

        self.pixelSize = ccdPol.get("pixelSize")

        nCol = ccdPol.get("nCol")
        nRow = ccdPol.get("nRow")
        if not ccdId:
            ccdId = cameraGeom.Id(ccdPol.get("serial"), ccdPol.get("name"))

        ccd = cameraGeom.Ccd(ccdId, self.pixelSize)

        if nCol*nRow != len(ccdPol.getArray("Amp")):
            raise RuntimeError, ("Expected location of %d amplifiers, got %d" % \
                                 (nCol*nRow, len(ccdPol.getArray("Amp"))))
        
        ampSerial0 = CameraGeomTestCase.ampSerial # used for testing
        for ampPol in ccdPol.getArray("Amp"):
            Col = ampPol.get("iCol")
            Row = ampPol.get("iRow")
            c =  ampPol.get("readoutCorner")

            if Col not in range(nCol) or Row not in range(nRow):
                raise RuntimeError, ("Amp location %d, %d is not in 0..%d, 0..%d" % (Col, Row, nCol, nRow))

            gain = ampPol.get("Electronic.gain")
            readNoise = ampPol.get("Electronic.readNoise")
            saturationLevel = ampPol.get("Electronic.saturationLevel")
            #
            # Now lookup properties common to all the CCD's amps
            #
            ampPol = self.geomPolicy.get("Amp")
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

            ampSerial = CameraGeomTestCase.ampSerial
            eParams = cameraGeom.ElectronicParams(gain, readNoise, saturationLevel)
            amp = cameraGeom.Amp(cameraGeom.Id(ampSerial, "ID%d" % ampSerial),
                                 allPixels, biasSec, dataSec, c, eParams)
            CameraGeomTestCase.ampSerial += 1
            
            ccd.addAmp(Col, Row, amp)
        #
        # Information for the test code
        #
        ccdInfo = {}
        ccdInfo["name"] = ccd.getId().getName()
        ccdInfo["ampWidth"], ccdInfo["ampHeight"] = width, height
        ccdInfo["width"], ccdInfo["height"] = nCol*eWidth, nRow*eHeight
        ccdInfo["trimmedWidth"], ccdInfo["trimmedHeight"] = nCol*width, nRow*height
        ccdInfo["ampIdMin"] = ampSerial0
        ccdInfo["ampIdMax"] = CameraGeomTestCase.ampSerial - 1
        #
        # Make an Image of that CCD?
        #
        if makeImage:
            ccdImage = afwImage.ImageU(ccd.getAllPixels().getDimensions())
            for a in ccd:
                im = ccdImage.Factory(ccdImage, a.getAllPixels())
                im += int(a.getElectronicParams().getReadNoise())
                im = ccdImage.Factory(ccdImage, a.getDataSec())
                im += int(1 + 100*a.getElectronicParams().getGain() + 0.5)
        else:
            ccdImage = None

        return ccd, ccdImage, ccdInfo

    def makeRaft(self, geomPolicy, raftId=None, makeImage=False):
        """Build a Raft from a set of CCDs given a suitable pex::Policy"""
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
            Col = ccdPol.get("iCol")
            Row = ccdPol.get("iRow")

            if Col not in range(nCol) or Row not in range(nRow):
                raise RuntimeError, ("Amp location %d, %d is not in 0..%d, 0..%d" % (Col, Row, nCol, nRow))

            ccdId = cameraGeom.Id(ccdPol.get("serial"), ccdPol.get("name"))
            ccd = self.makeCcd(geomPolicy, ccdId)[0]
            raft.addDetector(Col, Row, ccd)

        raftInfo = {}
        raftInfo["name"] = raft.getId().getName()
        raftInfo["width"] =  nCol*ccd.getAllPixels(True).getWidth()
        raftInfo["height"] = nRow*ccd.getAllPixels(True).getHeight()
        #
        # Make an Image of that Raft?
        #
        if makeImage:
            raftImage = afwImage.ImageU(raft.getAllPixels().getDimensions())
            for dl in raft:
                det = dl.getDetector();
                bbox = det.getAllPixels(True)
                bbox.shift(dl.getOrigin()[0], dl.getOrigin()[1])
                im = raftImage.Factory(raftImage, bbox)
                im += 1 + (det.getId().getSerial())
        else:
            raftImage = None

        return raft, raftImage, raftInfo

    def makeCamera(self, geomPolicy, cameraId=None, makeImage=False):
        """Build a Camera from a set of Rafts given a suitable pex::Policy"""
        cameraGeom.Camera = cameraGeom.DetectorMosaic

        cameraPol = geomPolicy.get("Camera")
        nCol = cameraPol.get("nCol")
        nRow = cameraPol.get("nRow")

        if not cameraId:
            cameraId = cameraGeom.Id(cameraPol.get("serial"), cameraPol.get("name"))
        camera = cameraGeom.Camera(cameraId)
        CameraGeomTestCase.cameraSerial += 1

        for ccdPol in cameraPol.getArray("Raft"):
            Col = ccdPol.get("iCol")
            Row = ccdPol.get("iRow")

            raftId = cameraGeom.Id(0, "R:%d,%d" % (Col, Row))
            raft = self.makeRaft(geomPolicy, raftId)[0]
            camera.addDetector(Col, Row, raft)

        cameraInfo = {}
        cameraInfo["name"] = camera.getId().getName()
        cameraInfo["width"] =  nCol*raft.getAllPixels(True).getWidth()
        cameraInfo["height"] = nRow*raft.getAllPixels(True).getHeight()
        #
        # Make an Image of that Camera?
        #
        if makeImage:
            cameraImage = afwImage.ImageU(camera.getAllPixels().getDimensions())
            for dl in camera:
                det = dl.getDetector();
                bbox = det.getAllPixels(True)
                bbox.shift(dl.getOrigin()[0], dl.getOrigin()[1])
                im = cameraImage.Factory(cameraImage, bbox)
                im += 1 + (det.getId().getSerial())
        else:
            cameraImage = None

        return camera, cameraImage, cameraInfo

    def setUp(self):
        CameraGeomTestCase.ampSerial = 0
        CameraGeomTestCase.ccdSerial = 1000
        CameraGeomTestCase.raftSerial = 2000
        CameraGeomTestCase.cameraSerial = 666

        policyFile = pexPolicy.DefaultPolicyFile("afw", "TestCameraGeomDictionary.paf", "tests")
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
        ccd, ccdImage, ccdInfo = self.makeCcd(self.geomPolicy, ccdId, makeImage=display)
        
        if display:
            showCcd(ccd, ccdImage, frame=0)

        for i in range(2):
            self.assertEqual(ccd.getSize()[i], self.pixelSize*ccd.getAllPixels(True).getDimensions()[i])

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
        trimmedImage = trimCcd(ccd, ccdImage)

        if display:
            showCcd(ccd, trimmedImage, frame=1)

        a = ccd.findAmp(cameraGeom.Id("ID%d" % ccdInfo["ampIdMin"]))
        self.assertEqual(a.getDataSec(), afwImage.BBox(afwImage.PointI(0, 0),
                                                       ccdInfo["ampWidth"], ccdInfo["ampHeight"]))

        self.assertEqual(ccd.getSize()[0], self.pixelSize*ccdInfo["trimmedWidth"])
        self.assertEqual(ccd.getSize()[1], self.pixelSize*ccdInfo["trimmedHeight"])
        #
        # Test mapping pixel <--> mm
        #
        pix = afwGeom.Point2I.makeXY(100, 200)
        pos = afwGeom.Point2D.makeXY(1.0, 2.0)
        
        self.assertEqualPoint(ccd.getIndexFromPosition(pos), pix)
        self.assertEqualPoint(ccd.getPositionFromIndex(pix), pos)

    def testRaft(self):
        """Test if we can build a Ccd out of Amps"""

        #print >> sys.stderr, "Skipping testRaft"; return

        raftId = cameraGeom.Id("Raft")
        raft, raftImage, raftInfo = self.makeRaft(self.geomPolicy, raftId, makeImage=True)

        if display:
            showRaft(raft, raftImage, frame=2)

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
            self.assertEqual(raft.getSize()[i], self.pixelSize*raft.getAllPixels().getDimensions()[i])
        
    def testCamera(self):
        """Test if we can build a Ccd out of Amps"""

        #print >> sys.stderr, "Skipping testCamera"; return

        camera, cameraImage, cameraInfo = self.makeCamera(self.geomPolicy, makeImage=True)

        if display:
            showCamera(camera, cameraImage, frame=3)

        if False:
            print "Camera Name \"%s\", serial %d,  BBox %s" % \
                  (camera.getId().getName(), camera.getId().getSerial(), camera.getAllPixels())

            for d in camera:
                print d.getOrigin(), d.getDetector().getAllPixels()

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
            self.assertEqual(camera.getSize()[i], self.pixelSize*camera.getAllPixels().getDimensions()[i])

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
