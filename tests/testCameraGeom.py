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
import tempfile

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept
import lsst.pex.policy as pexPolicy
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

import lsst.afw.table as afwTable
import lsst.afw.cameraGeom as cameraGeom
from lsst.afw.cameraGeom import DetectorConfig, CameraConfig, PIXELS, PUPIL, FOCAL_PLANE, CameraFactoryTask
import lsst.afw.cameraGeom.utils as cameraGeomUtils
try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
'''
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
'''
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def makeDetectorConfigs(detFile):
    detectors = []
    with open(detFile) as fh:
        names = fh.readline().rstrip().lstrip("#").split("|")
        for l in fh:
            els = l.rstrip().split("|")
            detectorProps = dict([(name, el) for name, el in zip(names, els)])
            detectors.append(detectorProps)
    detectorConfigs = []
    for detector in detectors:
        detConfig = DetectorConfig()
        detConfig.name = detector['name']
        detConfig.bbox_x0 = 0
        detConfig.bbox_y0 = 0
        detConfig.bbox_x1 = int(detector['npix_x']) - 1
        detConfig.bbox_y1 = int(detector['npix_y']) - 1
        detConfig.serial = str(detector['serial'])
        detConfig.detectorType = int(detector['detectorType'])
        detConfig.offset_x = float(detector['x'])
        detConfig.offset_y = float(detector['y'])
        detConfig.refpos_x = float(detector['refPixPos_x'])
        detConfig.refpos_y = float(detector['refPixPos_y'])
        detConfig.yawDeg = float(detector['yaw'])
        detConfig.pitchDeg = float(detector['pitch'])
        detConfig.rollDeg = float(detector['roll'])
        detConfig.pixelSize_x = float(detector['pixelSize'])
        detConfig.pixelSize_y = float(detector['pixelSize'])
        detConfig.transposeDetector = False
        detConfig.transformDict.nativeSys = PIXELS.getSysName()
        detectorConfigs.append(detConfig)
    return detectorConfigs

def makeAmpCatalogs(ampFile, isLsstLike=False):
    readoutMap = {'LL':0, 'LR':1, 'UR':2, 'UL':3}
    amps = []
    with open(ampFile) as fh:
        names = fh.readline().rstrip().lstrip("#").split("|")
        for l in fh:
            els = l.rstrip().split("|")
            ampProps = dict([(name, el) for name, el in zip(names, els)])
            amps.append(ampProps)
    ampTablesDict = {}
    schema = afwTable.AmpInfoTable.makeMinimalSchema()
    linThreshKey = schema.addField('linearityThreshold', type=float)
    linMaxKey = schema.addField('linearityMaximum', type=float)
    linUnitsKey = schema.addField('linearityUnits', type=str, size=9)
    for amp in amps:
        if amp['ccd_name'] in ampTablesDict:
            ampCatalog = ampTablesDict[amp['ccd_name']]
        else:
            ampCatalog = afwTable.AmpInfoCatalog(schema)
            ampTablesDict[amp['ccd_name']] = ampCatalog
        record = ampCatalog.addNew()
        bbox = afwGeom.Box2I(afwGeom.Point2I(int(amp['trimmed_xmin']), int(amp['trimmed_ymin'])),
                         afwGeom.Point2I(int(amp['trimmed_xmax']), int(amp['trimmed_ymax'])))
        rawBbox = afwGeom.Box2I(afwGeom.Point2I(int(amp['raw_xmin']), int(amp['raw_ymin'])),
                         afwGeom.Point2I(int(amp['raw_xmax']), int(amp['raw_ymax'])))
        rawDataBbox = afwGeom.Box2I(afwGeom.Point2I(int(amp['raw_data_xmin']), int(amp['raw_data_ymin'])),
                         afwGeom.Point2I(int(amp['raw_data_xmax']), int(amp['raw_data_ymax'])))
        rawHOverscanBbox = afwGeom.Box2I(afwGeom.Point2I(int(amp['hoscan_xmin']), int(amp['hoscan_ymin'])),
                         afwGeom.Point2I(int(amp['hoscan_xmax']), int(amp['hoscan_ymax'])))
        rawVOverscanBbox = afwGeom.Box2I(afwGeom.Point2I(int(amp['voscan_xmin']), int(amp['voscan_ymin'])),
                         afwGeom.Point2I(int(amp['voscan_xmax']), int(amp['voscan_ymax'])))
        rawPrescanBbox = afwGeom.Box2I(afwGeom.Point2I(int(amp['pscan_xmin']), int(amp['pscan_ymin'])),
                         afwGeom.Point2I(int(amp['pscan_xmax']), int(amp['pscan_ymax'])))
        xoffset = int(amp['x_offset'])
        yoffset = int(amp['y_offset'])
        flipx = bool(int(amp['flipx']))
        flipy = bool(int(amp['flipy']))
        readcorner = 'LL'
        if not isLsstLike:
            offext = afwGeom.Extent2I(xoffset, yoffset)
            if flipx:
                xExt = rawBbox.getDimensions().getX()
                rawBbox.flipLR(xExt)
                rawDataBbox.flipLR(xExt)
                rawHOverscanBbox.flipLR(xExt)
                rawVOverscanBbox.flipLR(xExt)
                rawPrescanBbox.flipLR(xExt)
            if flipy:
                yExt = rawBbox.getDimensions().getY()
                rawBbox.flipTB(yExt)
                rawDataBbox.flipTB(yExt)
                rawHOverscanBbox.flipTB(yExt)
                rawVOverscanBbox.flipTB(yExt)
                rawPrescanBbox.flipTB(yExt)
            if not flipx and not flipy:
                readcorner = 'LL'
            elif flipx and not flipy:
                readcorner = 'LR'
            elif flipx and flipy:
                readcorner = 'UR'
            elif not flipx and flipy:
                readcorner = 'UL'
            else:
                raise RuntimeError("Couldn't find read corner")

            flipx = False
            flipy = False
            rawBbox.shift(offext)
            rawDataBbox.shift(offext)
            rawHOverscanBbox.shift(offext)
            rawVOverscanBbox.shift(offext)
            rawPrescanBbox.shift(offext)
            xoffset = 0
            yoffset = 0
        offset = afwGeom.Extent2I(xoffset, yoffset)
        record.setBBox(bbox)
        record.setRawXYOffset(offset)
        record.setName(str(amp['name']))
        record.setReadoutCorner(readoutMap[readcorner])
        record.setGain(float(amp['gain']))
        record.setReadNoise(float(amp['readnoise']))
        record.setLinearityCoeffs([float(amp['lin_coeffs']),])
        record.setLinearityType(str(amp['lin_type']))
        record.setHasRawInfo(True)
        record.setRawFlipX(flipx)
        record.setRawFlipY(flipy)
        record.setRawBBox(rawBbox)
        record.setRawDataBBox(rawDataBbox)
        record.setRawHorizontalOverscanBBox(rawHOverscanBbox)
        record.setRawVerticalOverscanBBox(rawVOverscanBbox)
        record.setRawPrescanBBox(rawPrescanBbox)
        record.set(linThreshKey, float(amp['lin_thresh']))
        record.set(linMaxKey, float(amp['lin_max']))
        record.set(linUnitsKey, str(amp['lin_units']))
    return ampTablesDict

def makeTestRepositoryItems(isLsstLike=False):
    detFile = os.path.join(eups.productDir("afw"), "tests", "testCameraDetectors.dat")
    detectorConfigs = makeDetectorConfigs(detFile)
    ampFile = os.path.join(eups.productDir("afw"), "tests", "testCameraAmps.dat")
    ampCatalogDict = makeAmpCatalogs(ampFile, isLsstLike=isLsstLike)
    camConfig = CameraConfig()
    camConfig.detectorList = dict([(i,detectorConfigs[i]) for i in xrange(len(detectorConfigs))])
    plateScale = 20. #arcsec/mm
    camConfig.plateScale = plateScale
    pScaleRad = afwGeom.arcsecToRad(plateScale)
    #These came from the old test
    #radialDistortCoeffs = [0.0, 1.0/pScaleRad, 7.16417e-08, 3.03146e-10, 5.69338e-14, -6.61572e-18]
    #This matches what Dave M. has measured for an LSST like system.
    radialDistortCoeffs = [0.0, 1.0/pScaleRad, 0., 0.925/pScaleRad]
    tConfig = afwGeom.TransformConfig()
    tConfig.transform.name = 'radial'
    tConfig.transform.active.coeffs = radialDistortCoeffs
    tmc = afwGeom.TransformMapConfig()
    tmc.nativeSys = FOCAL_PLANE.getSysName()
    tmc.transforms = {PUPIL.getSysName():tConfig}
    camConfig.transformDict = tmc
    return camConfig, ampCatalogDict 
    

class CameraGeomTestCase(unittest.TestCase):
    """A test case for camera geometry"""

    def setUp(self):
        self.scCamConfig, self.scAmpCatalogDict = makeTestRepositoryItems()
        cameraTask = CameraFactoryTask()
        self.scCamera = cameraTask.runCatDict(self.scCamConfig, self.scAmpCatalogDict)
        self.lsstCamConfig, self.lsstAmpCatalogDict = makeTestRepositoryItems(isLsstLike=True)
        cameraTask = CameraFactoryTask()
        self.lsstCamera = cameraTask.runCatDict(self.lsstCamConfig, self.lsstAmpCatalogDict)

    def tearDown(self):
        del self.scCamera
        del self.lsstCamera
        del self.scCamConfig
        del self.scAmpCatalogDict
        del self.lsstCamConfig
        del self.lsstAmpCatalogDict

    def assertImagesAreEqual(self, outImage, compImage):
        """Assert that two images have all pixels equal"""
        self.assertTrue((outImage.getArray() == compImage.getArray()).all())

    def testConstructor(self):
        for camera in (self.scCamera, self.lsstCamera):
            self.assertTrue(isinstance(camera, Camera))
    def testMakeCameraPoint(self):
        
        for camera in (self.scCamera, self.lsstCamera):
            for coordSys in (PUPIL, FOCAL_PLANE):
                pt1 = afwGeom.Point2D(0.1, 0.3)
                pt2 = afwGeom.Point2D(0., 0.)
                pt3 = afwGeom.Point2D(-0.2, 0.2)
                pt4 = afwGeom.Point2D(0.02, -0.2)
                pt5 = afwGeom.Point2D(-0.2, -0.03)
                for pt in (pt1, pt2, pt3, pt4, pt5):
                    cp = camera.makeCameraPoint(pt, coordSys)
                    self.assertEquals(cp.getPoint(), pt)
                    self.assertEquals(cp.getCoordSys().getName(), coordSys.getName())

    def testTransform(self);
        #These test data come from SLALIB using SLA_PCD with 0.925 and
        #a plate scale of 20 arcsec/mm
        testData = [(-1.84000000, 1.04000000, -331.61689069, 187.43563387),
                    (-1.64000000, 0.12000000, -295.42491556, 21.61645724),
                    (-1.44000000, -0.80000000, -259.39818797, -144.11010443),
                    (-1.24000000, -1.72000000, -223.48275934, -309.99221457),
                    (-1.08000000, 1.36000000, -194.56520533, 245.00803635),
                    (-0.88000000, 0.44000000, -158.44320430, 79.22160215),
                    (-0.68000000, -0.48000000, -122.42389383, -86.41686623),
                    (-0.48000000, -1.40000000, -86.45332534, -252.15553224),
                    (-0.32000000, 1.68000000, -57.64746955, 302.64921514),
                    (-0.12000000, 0.76000000, -21.60360306, 136.82281940),
                    (0.08000000, -0.16000000, 14.40012984, -28.80025968),
                    (0.28000000, -1.08000000, 50.41767773, -194.46818554),
                    (0.48000000, -2.00000000, 86.50298919, -360.42912163),
                    (0.64000000, 1.08000000, 115.25115701, 194.48632746),
                    (0.84000000, 0.16000000, 151.23115189, 28.80593369),
                    (1.04000000, -0.76000000, 187.28751874, -136.86395600),
                    (1.24000000, -1.68000000, 223.47420612, -302.77150507),
                    (1.40000000, 1.40000000, 252.27834478, 252.27834478),
                    (1.60000000, 0.48000000, 288.22644118, 86.46793236),
                    (1.80000000, -0.44000000, 324.31346653, -79.27662515),]

        for camera in (self.scCamera, self.lsstCamera):
            for point in testData:
                fpTestPt = afwGeom.Point2D(point[2], point[3])
                pupilTestPt = afwGeom.Point2D(afwGeom.degToRad(point[0]), afwGeom.detToRad(point[1]0)
                cp = camera.makeCameraPoint(pupilTestPt, PUPIL)
                ncp = camera.transform(cp, FOCAL_PLANE)
                self.assertAlmostEquals(ncp.getPoint, fpTestPt)
                cp = camera.makeCameraPoint(fpTestPt, FOCAL_PLANE)
                ncp = camera.transform(cp, PUPIL)
                self.assertAlmostEquals(ncp.getPoint, pupilTestPt)
'''

    def testFindDetector(self):
        

    def testFpBbox(self):

    def testIteration(self):
        assert len = detectorConfigs.len

    def testDefaultTransform(self):

    def testCameraGeomUtils(self):

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

    def testLinearity(self):
        """Test if we can set Linearity parameters"""

        for ccdNum, threshold in [(-1, 0), (1234, 10),]:
            ccdId = cameraGeom.Id(ccdNum, "")
            ccd = cameraGeomUtils.makeCcd(self.geomPolicy, ccdId)
            for amp in list(ccd)[0:2]:  # only two amps in TestCameraGeom.paf
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
'''
