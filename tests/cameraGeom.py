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
import eups

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept
import lsst.pex.policy as pexPolicy
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.cameraGeom.utils as cameraGeomUtils
try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
class LsstLikeImage(cameraGeomUtils.GetCcdImage):
    def __init__(self, isTrimmed=True, isRaw=True):
        super(LsstLikeImage, self).__init__()
        self.isTrimmed = isTrimmed
        self.isRaw = isRaw
    def getImage(self, ccd, amp, imageFactory=afwImage.ImageU):
        im = imageFactory(os.path.join(eups.productDir("afw"), "tests",
            "test_amp.fits.gz"))
        if self.isTrimmed:
            bbox = amp.getElectronicDataSec()
        else:
            bbox = amp.getElectronicAllPixels()
        return amp.prepareAmpData(imageFactory(im, bbox, afwImage.LOCAL))

class ScLikeImage(cameraGeomUtils.GetCcdImage):
    def __init__(self, isTrimmed=True, isRaw=True):
        super(ScLikeImage, self).__init__()
        self.isTrimmed = isTrimmed
        self.isRaw = isRaw
    def getImage(self, ccd, amp, imageFactory=afwImage.ImageU):
        im = imageFactory(os.path.join(eups.productDir("afw"), "tests",
            "test.fits.gz"))
        if self.isTrimmed:
            bbox = amp.getDataSec()
        else:
            bbox = amp.getAllPixels()
        return imageFactory(im, bbox, afwImage.LOCAL)

def trimCcd(ccd, ccdImage=""):
    """Trim a Ccd and maybe the image of the untrimmed Ccd"""
    
    if ccdImage == "":
        ccdImage = cameraGeomUtils.makeImageFromCcd(ccd)

    if ccd.isTrimmed():
        return ccdImage

    ccd.setTrimmed(True)

    if ccdImage is not None:
        trimmedImage = ccdImage.Factory(ccd.getAllPixels())
        for a in ccd:
            data = ccdImage.Factory(ccdImage, a.getDataSec(False), afwImage.LOCAL)
            tdata = trimmedImage.Factory(trimmedImage, a.getDataSec(), afwImage.LOCAL)
            tdata <<= data
    else:
        trimmedImage = None
    """
    if trimmedImage:
        trimmedImage = afwMath.rotateImageBy90(trimmedImage, ccd.getOrientation().getNQuarter())
    """
    return trimmedImage

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class CameraGeomTestCase(unittest.TestCase):
    """A test case for camera geometry"""

    def setUp(self):
        CameraGeomTestCase.ampSerial = [0] # an array so we pass the value by reference

        self.geomPolicy = cameraGeomUtils.getGeomPolicy(os.path.join(eups.productDir("afw"),
                                                                     "tests", "TestCameraGeom.paf"))

    def tearDown(self):
        del self.geomPolicy

    def assertImagesAreEqual(self, outImage, compImage):
        """Assert that two images have all pixels equal"""
        self.assertTrue((outImage.getArray() == compImage.getArray()).all())

    def testDictionary(self):
        """Test the camera geometry dictionary"""

        if False:
            for r in self.geomPolicy.getArray("Raft"):
                print "raft", r
            for c in self.geomPolicy.getArray("Ccd"):
                print "ccd", c
            for a in self.geomPolicy.getArray("Amp"):
                print "amp", a

    def testRotatedCcd(self):
        """Test if we can build a Ccd out of Amps"""

        #print >> sys.stderr, "Skipping testRotatedCcd"; return

        ccdId = cameraGeom.Id(1, "Rot. CCD")
        ccdInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        ccd = cameraGeomUtils.makeCcd(self.geomPolicy, ccdId, ccdInfo=ccdInfo)
        zero = 0.0*afwGeom.radians
        ccd.setOrientation(cameraGeom.Orientation(1, zero, zero, zero))
        if display:
            cameraGeomUtils.showCcd(ccd)
            ds9.incrDefaultFrame()
        #
        # Trim the CCD and try again
        #


        trimmedImage = trimCcd(ccd)

        if display:
            cameraGeomUtils.showCcd(ccd, trimmedImage)
            ds9.incrDefaultFrame()

    def testId(self):
        """Test cameraGeom.Id"""

        #print >> sys.stderr, "Skipping testId"; return
        
        ix, iy = 2, 1
        id = cameraGeom.Id(666, "Beasty", ix, iy)
        self.assertTrue(id.getIndex(), (ix, iy))

        self.assertTrue(cameraGeom.Id(1) == cameraGeom.Id(1))
        self.assertFalse(cameraGeom.Id(1) == cameraGeom.Id(100))
        
        self.assertTrue(cameraGeom.Id("AA") == cameraGeom.Id("AA"))
        self.assertFalse(cameraGeom.Id("AA") == cameraGeom.Id("BB"))
        
        self.assertTrue(cameraGeom.Id(1, "AA") == cameraGeom.Id(1, "AA"))
        self.assertFalse(cameraGeom.Id(1, "AA") == cameraGeom.Id(2, "AA"))
        self.assertFalse(cameraGeom.Id(1, "AA") == cameraGeom.Id(1, "BB"))
        #
        self.assertTrue(cameraGeom.Id(1) < cameraGeom.Id(2))
        self.assertFalse(cameraGeom.Id(100) < cameraGeom.Id(1))
        
        self.assertTrue(cameraGeom.Id("AA") < cameraGeom.Id("BB"))
        
        self.assertTrue(cameraGeom.Id(1, "AA") < cameraGeom.Id(2, "AA"))
        self.assertTrue(cameraGeom.Id(1, "AA") < cameraGeom.Id(1, "BB"))

    def testSortedAmps(self):
        """Test if the Amps are sorted by ID after insertion into a Ccd"""

        ccd = cameraGeom.Ccd(cameraGeom.Id())
        Col = 0
        for serial in [0, 1, 2, 3, 4, 5, 6, 7]:
            gain, readNoise, saturationLevel = 0, 0, 0
            width, height = 10, 10

            allPixels = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(width, height))
            biasSec = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(0, height))
            dataSec = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(width, height))

            eParams = cameraGeom.ElectronicParams(gain, readNoise, saturationLevel)
            amp = cameraGeom.Amp(cameraGeom.Id(serial, "", Col, 0), allPixels, biasSec, dataSec,
                                 eParams)

            ccd.addAmp(afwGeom.Point2I(Col, 0), amp); Col += 1
        #
        # Check that Amps are sorted by Id
        #
        serials = []
        for a in ccd:
            serials.append(a.getId().getSerial())

        self.assertEqual(serials, sorted(serials))

    def testCcd(self):
        """Test if we can build a Ccd out of Amps"""

        #print >> sys.stderr, "Skipping testCcd"; return

        ccdId = cameraGeom.Id("CCD")
        ccdInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        ccd = cameraGeomUtils.makeCcd(self.geomPolicy, ccdId, ccdInfo=ccdInfo)
        if display:
            cameraGeomUtils.showCcd(ccd)
            ds9.incrDefaultFrame()
            trimmedImage = cameraGeomUtils.makeImageFromCcd(ccd, isTrimmed=True)
            cameraGeomUtils.showCcd(ccd, trimmedImage, isTrimmed=True)
            ds9.incrDefaultFrame()

        for i in range(2):
            self.assertEqual(ccd.getSize().getMm()[i],
                             ccdInfo["pixelSize"]*ccd.getAllPixels(True).getDimensions()[i])

        self.assertEqual(ccd.getId().getName(), ccdInfo["name"])
        self.assertEqual(ccd.getAllPixels().getWidth(), ccdInfo["width"])
        self.assertEqual(ccd.getAllPixels().getHeight(), ccdInfo["height"])
        self.assertEqual([a.getId().getSerial() for a in ccd],
                         range(ccdInfo["ampIdMin"], ccdInfo["ampIdMax"] + 1))

        id = cameraGeom.Id("ID%d" % ccdInfo["ampIdMax"])
        self.assertTrue(ccd.findAmp(id), id)

        self.assertEqual(ccd.findAmp(afwGeom.Point2I(10, 10)).getId().getSerial(), ccdInfo["ampIdMin"])

        self.assertEqual(ccd.getAllPixels().getMin(),
                         ccd.findAmp(afwGeom.Point2I(10, 10)).getAllPixels().getMin())

        self.assertEqual(ccd.getAllPixels().getMax(),
                         ccd.findAmp(afwGeom.Point2I(ccdInfo["width"] - 1,
                                                            ccdInfo["height"] - 1)).getAllPixels().getMax())
        ps = ccd.getPixelSize()
        #
        # Test mapping pixel <--> mm.  Use a pixel at the middle of the top of the CCD
        #
        pix = afwGeom.Point2D(99.5, 203.5)            # wrt bottom left
        pos = cameraGeom.FpPoint(0.00, 1.02)             # pixel center wrt CCD center
        posll = cameraGeom.FpPoint(0.00, 1.02)           # llc of pixel wrt CCD center
        #
        # Map pix into untrimmed coordinates
        #
        amp = ccd.findAmp(afwGeom.Point2I(int(pix[0]), int(pix[1])))
        corrI = amp.getDataSec(False).getMin() - amp.getDataSec(True).getMin()
        corr = afwGeom.Extent2D(corrI.getX(), corrI.getY())
        pix += corr
        
        self.assertEqual(amp.getDiskCoordSys(), cameraGeom.Amp.AMP)
        self.assertEqual(ccd.getPixelFromPosition(pos) + corr, pix)
        #
        # Trim the CCD and try again
        #
        trimmedImage = trimCcd(ccd)

        if display:
            ds9.mtv(trimmedImage, title='Trimmed')
            cameraGeomUtils.showCcd(ccd, trimmedImage)
            ds9.incrDefaultFrame()

        a = ccd.findAmp(cameraGeom.Id("ID%d" % ccdInfo["ampIdMin"]))
        self.assertEqual(a.getDataSec(), afwGeom.Box2I(afwGeom.Point2I(0, 0),
                                                       afwGeom.Extent2I(ccdInfo["ampWidth"], ccdInfo["ampHeight"])))

        self.assertEqual(ccd.getSize().getMm()[0], ccdInfo["pixelSize"]*ccdInfo["trimmedWidth"])
        self.assertEqual(ccd.getSize().getMm()[1], ccdInfo["pixelSize"]*ccdInfo["trimmedHeight"])
        #
        # Test mapping pixel <--> mm
        #
        pix = afwGeom.Point2D(99.5, 203.5)            # wrt bottom left
        pos = cameraGeom.FpPoint(0.00, 1.02)             # pixel center wrt CCD center
        posll = cameraGeom.FpPoint(0.00, 1.02)           # llc of pixel wrt CCD center
        
        self.assertEqual(ccd.getPixelFromPosition(pos), pix)
        self.assertEqual(ccd.getPositionFromPixel(pix).getMm(), posll.getMm())


    def testSortedCcds(self):
        """Test if the Ccds are sorted by ID after insertion into a Raft"""

        raft = cameraGeom.Raft(cameraGeom.Id(), 8, 1)
        Col = 0
        for serial in [7, 0, 1, 3, 2, 6, 5, 4]:
            ccd = cameraGeom.Ccd(cameraGeom.Id(serial))
            raft.addDetector(afwGeom.Point2I(Col, 0), cameraGeom.FpPoint(afwGeom.Point2D(0, 0)),
                             cameraGeom.Orientation(0), ccd)
            Col += 1
        #
        # Check that CCDs are sorted by Id
        #
        serials = []
        for ccd in raft:
            serials.append(ccd.getId().getSerial())

        self.assertEqual(serials, sorted(serials))

    def testRaft(self):
        """Test if we can build a Raft out of Ccds"""

        #print >> sys.stderr, "Skipping testRaft"; return
        raftId = cameraGeom.Id("Raft")
        raftInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        raft = cameraGeomUtils.makeRaft(self.geomPolicy, raftId, raftInfo=raftInfo)

        if display:
            cameraGeomUtils.showRaft(raft)
            ds9.incrDefaultFrame()

        if False:
            print cameraGeomUtils.describeRaft(raft)

        self.assertEqual(raft.getAllPixels().getWidth(), raftInfo["width"])
        self.assertEqual(raft.getAllPixels().getHeight(), raftInfo["height"])

        name = "C:0,2"
        self.assertEqual(raft.findDetector(cameraGeom.Id(name)).getId().getName(), name)

        self.assertEqual(raft.getSize().getMm()[0], raftInfo["widthMm"])
        self.assertEqual(raft.getSize().getMm()[1], raftInfo["heightMm"])
        #
        # Test mapping pixel <--> mm
        #
        ps = raft.getPixelSize()
        for ix, iy, x, y in [(102, 500, -1.01,  2.02),
                             (306, 100,  1.01, -2.02),
                             (306, 500,  1.01,  2.02),
                             (356, 525,  1.51,  2.27),
                             ]:
            pix = afwGeom.Point2I(ix, iy) # wrt raft LLC
            #position of pixel center
            pos = cameraGeom.FpPoint(x+ps/2., y+ps/2.) # wrt raft center
            #position of pixel lower left corner which is returned by getPositionFromPixel()
            posll = cameraGeom.FpPoint(x, y) # wrt raft center

            rpos = raft.getPixelFromPosition(pos)
            rpos = afwGeom.PointI(int(rpos.getX()), int(rpos.getY()))
            # need to rework cameraGeom since FpPoint changes.  disable this for now
            if False:
                self.assertEqual(rpos, pix)

            # this test is no longer meaningful as pixel is specific to a detector xy0
            if False:
                self.assertEqual(raft.getPositionFromPixel(afwGeom.Point2D(pix[0], pix[1])).getMm(),
                                 posll.getMm())
                
        
    def testCamera(self):
        """Test if we can build a Camera out of Rafts"""

        #print >> sys.stderr, "Skipping testCamera"; return

        cameraInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        camera = cameraGeomUtils.makeCamera(self.geomPolicy, cameraInfo=cameraInfo)

        if display:
            cameraGeomUtils.showCamera(camera, )
            ds9.incrDefaultFrame()

        if False:
            print cameraGeomUtils.describeCamera(camera)

        self.assertEqual(camera.getAllPixels().getWidth(), cameraInfo["width"])
        self.assertEqual(camera.getAllPixels().getHeight(), cameraInfo["height"])

        name = "R:1,0"
        self.assertEqual(camera.findDetector(cameraGeom.Id(name)).getId().getName(), name)

        self.assertEqual(camera.getSize().getMm()[0], cameraInfo["widthMm"])
        self.assertEqual(camera.getSize().getMm()[1], cameraInfo["heightMm"])

        #
        # Test mapping pixel <--> mm
        #
        for ix, iy, x, y in [(102, 500, -3.12, 2.02),
                             (152, 525, -2.62, 2.27),
                             (714, 500,  3.12, 2.02),
                             ]:
            pix = afwGeom.PointD(ix, iy) # wrt raft LLC
            pos = cameraGeom.FpPoint(x, y) # center of pixel wrt raft center
            posll = cameraGeom.FpPoint(x, y) # llc of pixel wrt raft center

            # may need to restructure this since adding FpPoint
            if False:
                self.assertEqual(camera.getPixelFromPosition(pos), pix)

            # there is no unique mapping from a pixel to a focal plane position
            #  ... the pixel could be on any ccd
            if False:
                self.assertEqual(camera.getPositionFromPixel(pix).getMm(), posll.getMm())
            
        # Check that we can find an Amp in the bowels of the camera
        ccdName = "C:1,0"
        amp = cameraGeomUtils.findAmp(camera, cameraGeom.Id(ccdName), 1, 2)
        self.assertFalse(amp is None)
        self.assertEqual(amp.getId().getName(), "ID7")
        self.assertEqual(amp.getParent().getId().getName(), ccdName)

    def testDefectBase(self):
        """Test DefectBases"""

        #print >> sys.stderr, "Skipping testDefectBase"; return

        defectsDict = cameraGeomUtils.makeDefects(self.geomPolicy)

        for ccdName in ("Defective", "Defective II"):
            ccd = cameraGeomUtils.makeCcd(self.geomPolicy, cameraGeom.Id(ccdName))

            ccdImage = cameraGeomUtils.makeImageFromCcd(ccd)

            if ccdName == "Defective":
                #
                # Insert some defects into the Ccd
                #
                for x0, y0, x1, y1 in [
                    (34,  0,   35,  80 ),
                    (34,  81,  34,  100),
                    (180, 100, 182, 130),
                    ]:
                    bbox = afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Point2I(x1, y1))
                    bad = ccdImage.Factory(ccdImage, bbox, afwImage.LOCAL)
                    bad.set(100)

                if display:
                    ds9.mtv(ccdImage, title="Defects")
                    cameraGeomUtils.showCcd(ccd, None)

            defects = [v for (k, v) in defectsDict.items() if k == ccd.getId()]
            if len(defects) == 0:
                contine
            elif len(defects) == 1:
                defects = defects[0]
            else:
                raise RuntimeError, ("Found more than one defect set for CCD %s" % ccd.getId())

            ccd.setDefects(defects)

            if False:
                print "CCD (%s)" % ccd.getId()

                for a in ccd:
                    print "    ", a.getId(), [str(d.getBBox()) for d in a.getDefects()]

            if ccdName == "Defective" and display:
                for d in ccd.getDefects():
                    displayUtils.drawBBox(d.getBBox(), ctype=ds9.CYAN, borderWidth=1.5)

                for a in ccd:
                    for d in a.getDefects():
                        displayUtils.drawBBox(d.getBBox(), ctype=ds9.YELLOW, borderWidth=1.0)

                ds9.incrDefaultFrame()

    def testAssembleCcd(self):
        """Test if we can build a Ccd out of Amps"""

        compImage = afwImage.ImageU(os.path.join(eups.productDir("afw"),
                                                 "tests", "test_comp.fits.gz"))
        compImageTrimmed = afwImage.ImageU(os.path.join(eups.productDir("afw"), "tests",
                                                        "test_comp_trimmed.fits.gz"))

        ccdId = cameraGeom.Id(1, "LsstLike")
        ccdInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        ccd = cameraGeomUtils.makeCcd(self.geomPolicy, ccdId, ccdInfo=ccdInfo)
        #
        # Test assembly of images that require preparation for assembly (like
        # LSST images)
        #
        outImage = cameraGeomUtils.makeImageFromCcd(ccd,
                    imageSource=LsstLikeImage(),
                    isTrimmed=False, imageFactory=afwImage.ImageU)

        self.assertImagesAreEqual(outImage, compImage)

        if display:
            cameraGeomUtils.showCcd(ccd, outImage)
            ds9.incrDefaultFrame()

        #
        # Test assembly of images that reside in a pre-assembled state from
        # the DAQ (like Suprime-Cam images)
        #

        ccdId = cameraGeom.Id(1, "ScLike")
        ccdInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        ccd = cameraGeomUtils.makeCcd(self.geomPolicy, ccdId, ccdInfo=ccdInfo)
        
        outImage = cameraGeomUtils.makeImageFromCcd(ccd,
                    imageSource=ScLikeImage(),
                    isTrimmed=False, imageFactory=afwImage.ImageU)

        self.assertImagesAreEqual(outImage, compImage)

        if display:
            cameraGeomUtils.showCcd(ccd, outImage)
            ds9.incrDefaultFrame()

        #
        # Do the same tests for trimmed ccds.
        #
        ccdId = cameraGeom.Id(1, "LsstLike")
        ccdInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        ccd = cameraGeomUtils.makeCcd(self.geomPolicy, ccdId, ccdInfo=ccdInfo)

        outImage = cameraGeomUtils.makeImageFromCcd(ccd,
                    imageSource=LsstLikeImage(),
                    isTrimmed=True, imageFactory=afwImage.ImageU)
        ccd.setTrimmed(True)
        self.assertImagesAreEqual(outImage, compImageTrimmed)

        if display:
            cameraGeomUtils.showCcd(ccd, outImage)
            ds9.incrDefaultFrame()

        ccdId = cameraGeom.Id(1, "ScLike")
        ccdInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        ccd = cameraGeomUtils.makeCcd(self.geomPolicy, ccdId, ccdInfo=ccdInfo)

        outImage = cameraGeomUtils.makeImageFromCcd(ccd,
                    imageSource=ScLikeImage(),
                    isTrimmed=True, imageFactory=afwImage.ImageU)
        ccd.setTrimmed(True)
        self.assertImagesAreEqual(outImage, compImageTrimmed)

        if display:
            cameraGeomUtils.showCcd(ccd, outImage)
            ds9.incrDefaultFrame()

    def testAddTrimmedAmp(self):
        """Test that building a Ccd from trimmed Amps leads to a trimmed Ccd"""

        dataSec = afwGeom.BoxI(afwGeom.PointI(1, 0), afwGeom.ExtentI(10, 20))
        biasSec = afwGeom.BoxI(afwGeom.PointI(0, 0), afwGeom.ExtentI(1, 1))
        allPixelsInAmp = afwGeom.BoxI(afwGeom.PointI(0, 0), afwGeom.ExtentI(11, 21))
        eParams = cameraGeom.ElectronicParams(1.0, 1.0, 65000)

        ccd = cameraGeom.Ccd(cameraGeom.Id(0))
        self.assertFalse(ccd.isTrimmed())

        for i in range(2):
            amp = cameraGeom.Amp(cameraGeom.Id(i, "", i, 0),
                                 allPixelsInAmp, biasSec, dataSec, eParams)
            amp.setTrimmed(True)

            if i%2 == 0:                # check both APIs
                ccd.addAmp(afwGeom.PointI(i, 0), amp)
            else:
                ccd.addAmp(amp)
            self.assertTrue(ccd.isTrimmed())

        # should fail to add non-trimmed Amp to a trimmed Ccd
        i += 1
        amp = cameraGeom.Amp(cameraGeom.Id(i, "", i, 0),
                             allPixelsInAmp, biasSec, dataSec, eParams)
        self.assertFalse(amp.isTrimmed())

        utilsTests.assertRaisesLsstCpp(self, pexExcept.InvalidParameterException, ccd.addAmp, amp)

        # should fail to add trimmed Amp to a non-trimmed Ccd
        ccd.setTrimmed(False)
        amp.setTrimmed(True)
        utilsTests.assertRaisesLsstCpp(self, pexExcept.InvalidParameterException, ccd.addAmp, amp)
        
    def testLinearity(self):
        """Test if we can set Linearity parameters"""

        for ccdNum, threshold in [(-1, 0), (1234, 10),]:
            ccdId = cameraGeom.Id(ccdNum, "")
            ccd = cameraGeomUtils.makeCcd(self.geomPolicy, ccdId)
            amp = list(ccd)[0]
            lin = amp.getElectronicParams().getLinearity()
            self.assertEqual(lin.threshold, threshold)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    if display:
        ds9.cmdBuffer.pushSize()

    suites = []
    suites += unittest.makeSuite(CameraGeomTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    if display:
        ds9.cmdBuffer.popSize()

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""

    if display:
        ds9.setDefaultFrame(0)
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
