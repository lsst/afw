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
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.cameraGeom.utils as cameraGeomUtils

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def trimCcd(ccd, ccdImage=""):
    """Trim a Ccd and maybe the image of the untrimmed Ccd"""
    
    if ccdImage == "":
        ccdImage = cameraGeomUtils.makeImageFromCcd(ccd)

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

    def setUp(self):
        CameraGeomTestCase.ampSerial = [0] # an array so we pass the value by reference

        self.geomPolicy = cameraGeomUtils.getGeomPolicy(os.path.join(eups.productDir("afw"),
                                                                     "tests", "TestCameraGeom.paf"))

    def tearDown(self):
        del self.geomPolicy

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

            allPixels = afwImage.BBox(afwImage.PointI(0, 0), width, height)
            biasSec = afwImage.BBox(afwImage.PointI(0, 0), 0, height)
            dataSec = afwImage.BBox(afwImage.PointI(0, 0), width, height)

            eParams = cameraGeom.ElectronicParams(gain, readNoise, saturationLevel)
            amp = cameraGeom.Amp(cameraGeom.Id(serial, "", Col, 0), allPixels, biasSec, dataSec,
                                 cameraGeom.Amp.LLC, eParams)

            ccd.addAmp(afwGeom.makePointI(Col, 0), amp); Col += 1
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

        for i in range(2):
            self.assertEqual(ccd.getSize()[i], ccdInfo["pixelSize"]*ccd.getAllPixels(True).getDimensions()[i])

        self.assertEqual(ccd.getId().getName(), ccdInfo["name"])
        self.assertEqual(ccd.getAllPixels().getWidth(), ccdInfo["width"])
        self.assertEqual(ccd.getAllPixels().getHeight(), ccdInfo["height"])
        self.assertEqual([a.getId().getSerial() for a in ccd],
                         range(ccdInfo["ampIdMin"], ccdInfo["ampIdMax"] + 1))

        id = cameraGeom.Id("ID%d" % ccdInfo["ampIdMax"])
        self.assertTrue(ccd.findAmp(id), id)

        self.assertEqual(ccd.findAmp(afwGeom.makePointI(10, 10)).getId().getSerial(), ccdInfo["ampIdMin"])

        self.assertEqual(ccd.getAllPixels().getLLC(),
                         ccd.findAmp(afwGeom.makePointI(10, 10)).getAllPixels().getLLC())

        self.assertEqual(ccd.getAllPixels().getURC(),
                         ccd.findAmp(afwGeom.makePointI(ccdInfo["width"] - 1,
                                                            ccdInfo["height"] - 1)).getAllPixels().getURC())
        ps = ccd.getPixelSize()
        #
        # Test mapping pixel <--> mm
        #
        pix = afwGeom.makePointI(100, 204) # wrt bottom left
        pos = afwGeom.makePointD(0.00+ps/2., 1.02+ps/2.) # pixel center wrt CCD center
        posll = afwGeom.makePointD(0.00, 1.02) # llc of pixel wrt CCD center
        #
        # Map pix into untrimmed coordinates
        #
        amp = ccd.findAmp(pix)
        corr = amp.getDataSec(False).getLLC() - amp.getDataSec(True).getLLC()
        corr = afwGeom.Extent2I(afwGeom.makePointI(corr[0], corr[1]))
        pix += corr
        
        self.assertEqual(ccd.getPixelFromPosition(pos) + corr, pix)
        self.assertEqual(ccd.getPositionFromPixel(pix), posll)
        #
        # Trim the CCD and try again
        #
        trimmedImage = trimCcd(ccd)

        if display:
            ds9.mtv(trimmedImage, title='Trimmed')
            cameraGeomUtils.showCcd(ccd, trimmedImage)
            ds9.incrDefaultFrame()

        a = ccd.findAmp(cameraGeom.Id("ID%d" % ccdInfo["ampIdMin"]))
        self.assertEqual(a.getDataSec(), afwImage.BBox(afwImage.PointI(0, 0),
                                                       ccdInfo["ampWidth"], ccdInfo["ampHeight"]))

        self.assertEqual(ccd.getSize()[0], ccdInfo["pixelSize"]*ccdInfo["trimmedWidth"])
        self.assertEqual(ccd.getSize()[1], ccdInfo["pixelSize"]*ccdInfo["trimmedHeight"])
        #
        # Test mapping pixel <--> mm
        #
        pix = afwGeom.makePointI(100, 204) # wrt LLC
        pos = afwGeom.makePointD(0.00+ps/2., 1.02+ps/2.) #pixel center wrt CCD center
        posll = afwGeom.makePointD(0.0, 1.02) # llc of pixel wrt chip center 
        
        self.assertEqual(ccd.getPixelFromPosition(pos), pix)
        self.assertEqual(ccd.getPositionFromPixel(pix), posll)

    def testRotatedCcd(self):
        """Test if we can build a Ccd out of Amps"""

        #print >> sys.stderr, "Skipping testRotatedCcd"; return

        ccdId = cameraGeom.Id("Rotated CCD")
        ccdInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        ccd = cameraGeomUtils.makeCcd(self.geomPolicy, ccdId, ccdInfo=ccdInfo)
        ccd.setOrientation(cameraGeom.Orientation(1, 0.0, 0.0, 0.0))
        if display:
            cameraGeomUtils.showCcd(ccd)
            ds9.incrDefaultFrame()
        #
        # Trim the CCD and try again
        #
        trimmedImage = trimCcd(ccd)

        if display:
            ds9.mtv(trimmedImage, title='Rotated trimmed')
            cameraGeomUtils.showCcd(ccd, trimmedImage)
            ds9.incrDefaultFrame()

    def testSortedCcds(self):
        """Test if the Ccds are sorted by ID after insertion into a Raft"""

        raft = cameraGeom.Raft(cameraGeom.Id(), 8, 1)
        Col = 0
        for serial in [7, 0, 1, 3, 2, 6, 5, 4]:
            ccd = cameraGeom.Ccd(cameraGeom.Id(serial))
            raft.addDetector(afwGeom.makePointI(Col, 0), afwGeom.makePointD(0, 0),
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

        for x, y, serial, cen in [(  0,   0,  5, (-1.01, -2.02)),
                                  (150, 250, 21, (-1.01,  0.0 )),
                                  (250, 250, 29, ( 1.01,  0.0 )),
                                  (300, 500, 42, ( 1.01,  2.02))]:
            det = raft.findDetector(afwGeom.makePointI(x, y))
            ccd = cameraGeom.cast_Ccd(det)
            if False:
                print x, y, det.getId().getName(), \
                      ccd.findAmp(afwGeom.makePointI(150, 152), True).getId().getSerial()
            self.assertEqual(ccd.findAmp(afwGeom.makePointI(150, 152), True).getId().getSerial(), serial)
            for i in range(2):
                self.assertAlmostEqual(ccd.getCenter()[i], cen[i])

        name = "C:0,2"
        self.assertEqual(raft.findDetector(cameraGeom.Id(name)).getId().getName(), name)

        self.assertEqual(raft.getSize()[0], raftInfo["widthMm"])
        self.assertEqual(raft.getSize()[1], raftInfo["heightMm"])
        #
        # Test mapping pixel <--> mm
        #
        ps = raft.getPixelSize()
        for ix, iy, x, y in [(102, 500, -1.01,  2.02),
                             (306, 100,  1.01, -2.02),
                             (306, 500,  1.01,  2.02),
                             (356, 525,  1.51,  2.27),
                             ]:
            pix = afwGeom.makePointI(ix, iy) # wrt raft LLC
            #position of pixel center
            pos = afwGeom.makePointD(x+ps/2., y+ps/2.) # wrt raft center
            #position of pixel lower left corner which is returned by getPositionFromPixel()
            posll = afwGeom.makePointD(x, y) # wrt raft center

            self.assertEqual(raft.getPixelFromPosition(pos), pix)
            self.assertEqual(raft.getPositionFromPixel(pix), posll)
        
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

        for rx, ry, cx, cy, serial, cen in [(0, 0,     0,   0,    4, (-3.12, -2.02)),
                                            (0,   0,   150, 250, 20, (-3.12,  0.00)),
                                            (600, 300, 0,   0,   52, ( 1.10,  -2.02)),
                                            (600, 300, 150, 250, 68, ( 1.10,  0.00)),
                                            ]:
            raft = cameraGeom.cast_Raft(camera.findDetector(afwGeom.makePointI(rx, ry)))

            ccd = cameraGeom.cast_Ccd(raft.findDetector(afwGeom.makePointI(cx, cy)))
            self.assertEqual(ccd.findAmp(afwGeom.makePointI(153, 152), True).getId().getSerial(), serial)
            for i in range(2):
                self.assertAlmostEqual(ccd.getCenter()[i], cen[i])

        name = "R:1,0"
        self.assertEqual(camera.findDetector(cameraGeom.Id(name)).getId().getName(), name)

        self.assertEqual(camera.getSize()[0], cameraInfo["widthMm"])
        self.assertEqual(camera.getSize()[1], cameraInfo["heightMm"])
        ps = raft.getPixelSize()
        #
        # Test mapping pixel <--> mm
        #
        for ix, iy, x, y in [(102, 500, -3.12, 2.02),
                             (152, 525, -2.62, 2.27),
                             (714, 500,  3.12, 2.02),
                             ]:
            pix = afwGeom.makePointI(ix, iy) # wrt raft LLC
            pos = afwGeom.makePointD(x+ps/2., y+ps/2.) # center of pixel wrt raft center
            posll = afwGeom.makePointD(x, y) # llc of pixel wrt raft center
            
            self.assertEqual(camera.getPixelFromPosition(pos), pix)
            self.assertEqual(camera.getPositionFromPixel(pix), posll)
        # Check that we can find an Amp in the bowels of the camera
        ccdName = "C:0,0"
        amp = cameraGeomUtils.findAmp(camera, cameraGeom.Id(ccdName), 1, 2)
        self.assertEqual(amp.getId().getName(), "ID6")
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
                    bbox = afwImage.BBox(afwImage.PointI(x0, y0), afwImage.PointI(x1, y1))
                    bad = ccdImage.Factory(ccdImage, bbox)
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

    def testParent(self):
        """Test that we can find our parent"""

        cameraInfo = {"ampSerial" : CameraGeomTestCase.ampSerial}
        camera = cameraGeomUtils.makeCamera(self.geomPolicy, cameraInfo=cameraInfo)

        for rx, ry, cx, cy, serial in [(0, 0,     0,   0,   4),
                                       (0,   0,   150, 250, 20),
                                       (600, 300, 0,   0,   52),
                                       (600, 300, 150, 250, 68),
                                       ]:
            raft = cameraGeom.cast_Raft(camera.findDetector(afwGeom.makePointI(rx, ry)))
            ccd = cameraGeom.cast_Ccd(raft.findDetector(afwGeom.makePointI(cx, cy)))

            amp = ccd[0]
            self.assertEqual(ccd.getId(),    amp.getParent().getId())
            self.assertEqual(raft.getId(),   ccd.getParent().getId())
            self.assertEqual(camera.getId(), ccd.getParent().getParent().getId())
            self.assertEqual(None,           ccd.getParent().getParent().getParent())        

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

    ds9.setDefaultFrame(0)
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
