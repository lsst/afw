# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Test lsst.afw.image.Exposure
"""

import dataclasses
import os.path
import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose
import yaml
import astropy.units as units

import lsst.utils
import lsst.utils.tests
import lsst.geom
import lsst.afw.image as afwImage
from lsst.afw.coord import Weather
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
import lsst.pex.exceptions as pexExcept
from lsst.afw.fits import readMetadata, FitsError
from lsst.afw.cameraGeom.testUtils import DetectorWrapper
from lsst.daf.base import PropertyList
from lsst.log import Log
from testTableArchivesLib import DummyPsf

Log.getLogger("lsst.afw.image.Mask").setLevel(Log.INFO)

try:
    dataDir = os.path.join(lsst.utils.getPackageDir("afwdata"), "data")
except LookupError:
    dataDir = None
else:
    InputMaskedImageName = "871034p_1_MI.fits"
    InputMaskedImageNameSmall = "small_MI.fits"
    InputImageNameSmall = "small"
    OutputMaskedImageName = "871034p_1_MInew.fits"

    currDir = os.path.abspath(os.path.dirname(__file__))
    inFilePath = os.path.join(dataDir, InputMaskedImageName)
    inFilePathSmall = os.path.join(dataDir, InputMaskedImageNameSmall)
    inFilePathSmallImage = os.path.join(dataDir, InputImageNameSmall)


@unittest.skipIf(dataDir is None, "afwdata not setup")
class ExposureTestCase(lsst.utils.tests.TestCase):
    """
    A test case for the Exposure Class
    """

    def setUp(self):
        maskedImage = afwImage.MaskedImageF(inFilePathSmall)
        maskedImageMD = readMetadata(inFilePathSmall)

        self.smallExposure = afwImage.ExposureF(inFilePathSmall)
        self.width = maskedImage.getWidth()
        self.height = maskedImage.getHeight()
        self.wcs = afwGeom.makeSkyWcs(maskedImageMD, False)
        self.md = maskedImageMD
        self.psf = DummyPsf(2.0)
        self.detector = DetectorWrapper().detector
        self.id = 42
        self.extras = {"MISC": DummyPsf(3.5)}

        self.exposureBlank = afwImage.ExposureF()
        self.exposureMiOnly = afwImage.makeExposure(maskedImage)
        self.exposureMiWcs = afwImage.makeExposure(maskedImage, self.wcs)
        # n.b. the (100, 100, ...) form
        self.exposureCrWcs = afwImage.ExposureF(100, 100, self.wcs)
        # test with ExtentI(100, 100) too
        self.exposureCrOnly = afwImage.ExposureF(lsst.geom.ExtentI(100, 100))

    def tearDown(self):
        del self.smallExposure
        del self.wcs
        del self.psf
        del self.detector
        del self.extras

        del self.exposureBlank
        del self.exposureMiOnly
        del self.exposureMiWcs
        del self.exposureCrWcs
        del self.exposureCrOnly

    def testGetMaskedImage(self):
        """
        Test to ensure a MaskedImage can be obtained from each
        Exposure. An Exposure is required to have a MaskedImage,
        therefore each of the Exposures should return a MaskedImage.

        MaskedImage class should throw appropriate
        lsst::pex::exceptions::NotFound if the MaskedImage can not be
        obtained.
        """
        maskedImageBlank = self.exposureBlank.getMaskedImage()
        blankWidth = maskedImageBlank.getWidth()
        blankHeight = maskedImageBlank.getHeight()
        if blankWidth != blankHeight != 0:
            self.fail(f"{blankWidth} = {blankHeight} != 0")

        maskedImageMiOnly = self.exposureMiOnly.getMaskedImage()
        miOnlyWidth = maskedImageMiOnly.getWidth()
        miOnlyHeight = maskedImageMiOnly.getHeight()
        self.assertAlmostEqual(miOnlyWidth, self.width)
        self.assertAlmostEqual(miOnlyHeight, self.height)

        # NOTE: Unittests for Exposures created from a MaskedImage and
        # a WCS object are incomplete.  No way to test the validity of
        # the WCS being copied/created.

        maskedImageMiWcs = self.exposureMiWcs.getMaskedImage()
        miWcsWidth = maskedImageMiWcs.getWidth()
        miWcsHeight = maskedImageMiWcs.getHeight()
        self.assertAlmostEqual(miWcsWidth, self.width)
        self.assertAlmostEqual(miWcsHeight, self.height)

        maskedImageCrWcs = self.exposureCrWcs.getMaskedImage()
        crWcsWidth = maskedImageCrWcs.getWidth()
        crWcsHeight = maskedImageCrWcs.getHeight()
        if crWcsWidth != crWcsHeight != 0:
            self.fail(f"{crWcsWidth} != {crWcsHeight} != 0")

        maskedImageCrOnly = self.exposureCrOnly.getMaskedImage()
        crOnlyWidth = maskedImageCrOnly.getWidth()
        crOnlyHeight = maskedImageCrOnly.getHeight()
        if crOnlyWidth != crOnlyHeight != 0:
            self.fail(f"{crOnlyWidth} != {crOnlyHeight} != 0")

        # Check Exposure.getWidth() returns the MaskedImage's width
        self.assertEqual(crOnlyWidth, self.exposureCrOnly.getWidth())
        self.assertEqual(crOnlyHeight, self.exposureCrOnly.getHeight())
        # check width/height properties
        self.assertEqual(crOnlyWidth, self.exposureCrOnly.width)
        self.assertEqual(crOnlyHeight, self.exposureCrOnly.height)

    def testProperties(self):
        self.assertMaskedImagesEqual(self.exposureMiOnly.maskedImage,
                                     self.exposureMiOnly.getMaskedImage())
        mi2 = afwImage.MaskedImageF(self.exposureMiOnly.getDimensions())
        mi2.image.array[:] = 5.0
        mi2.variance.array[:] = 3.0
        mi2.mask.array[:] = 0x1
        self.exposureMiOnly.maskedImage = mi2
        self.assertMaskedImagesEqual(self.exposureMiOnly.maskedImage, mi2)
        self.assertImagesEqual(self.exposureMiOnly.image,
                               self.exposureMiOnly.maskedImage.image)

        image3 = afwImage.ImageF(self.exposureMiOnly.getDimensions())
        image3.array[:] = 3.0
        self.exposureMiOnly.image = image3
        self.assertImagesEqual(self.exposureMiOnly.image, image3)

        mask3 = afwImage.MaskX(self.exposureMiOnly.getDimensions())
        mask3.array[:] = 0x2
        self.exposureMiOnly.mask = mask3
        self.assertMasksEqual(self.exposureMiOnly.mask, mask3)

        var3 = afwImage.ImageF(self.exposureMiOnly.getDimensions())
        var3.array[:] = 2.0
        self.exposureMiOnly.variance = var3
        self.assertImagesEqual(self.exposureMiOnly.variance, var3)

        # Test the property getter for a null VisitInfo.
        self.assertIsNone(self.exposureMiOnly.visitInfo)

    def testGetWcs(self):
        """Test that a WCS can be obtained from each Exposure created with
        a WCS, and that an Exposure lacking a WCS returns None.
        """
        # These exposures don't contain a WCS
        self.assertIsNone(self.exposureBlank.getWcs())
        self.assertIsNone(self.exposureMiOnly.getWcs())
        self.assertIsNone(self.exposureCrOnly.getWcs())

        # These exposures should contain a WCS
        self.assertEqual(self.wcs, self.exposureMiWcs.getWcs())
        self.assertEqual(self.wcs, self.exposureCrWcs.getWcs())

    def testExposureInfoConstructor(self):
        """Test the Exposure(maskedImage, exposureInfo) constructor"""
        exposureInfo = afwImage.ExposureInfo()
        exposureInfo.setWcs(self.wcs)
        exposureInfo.setDetector(self.detector)
        gFilterLabel = afwImage.FilterLabel(band="g")
        exposureInfo.setFilter(gFilterLabel)
        maskedImage = afwImage.MaskedImageF(inFilePathSmall)
        exposure = afwImage.ExposureF(maskedImage, exposureInfo)

        self.assertTrue(exposure.hasWcs())
        self.assertEqual(exposure.getWcs().getPixelOrigin(),
                         self.wcs.getPixelOrigin())
        self.assertEqual(exposure.getDetector().getName(),
                         self.detector.getName())
        self.assertEqual(exposure.getDetector().getSerial(),
                         self.detector.getSerial())
        self.assertEqual(exposure.getFilter(), gFilterLabel)

        self.assertTrue(exposure.getInfo().hasWcs())
        # check the ExposureInfo property
        self.assertTrue(exposure.info.hasWcs())
        self.assertEqual(exposure.getInfo().getWcs().getPixelOrigin(),
                         self.wcs.getPixelOrigin())
        self.assertEqual(exposure.getInfo().getDetector().getName(),
                         self.detector.getName())
        self.assertEqual(exposure.getInfo().getDetector().getSerial(),
                         self.detector.getSerial())
        self.assertEqual(exposure.getInfo().getFilter(), gFilterLabel)

    def testNullWcs(self):
        """Test that an Exposure constructed with second argument None is usable

        When the exposureInfo constructor was first added, trying to get a WCS
        or other info caused a segfault because the ExposureInfo did not exist.
        """
        maskedImage = self.exposureMiOnly.getMaskedImage()
        exposure = afwImage.ExposureF(maskedImage, None)
        self.assertFalse(exposure.hasWcs())
        self.assertFalse(exposure.hasPsf())

    def testExposureInfoSetNone(self):
        exposureInfo = afwImage.ExposureInfo()
        exposureInfo.setDetector(None)
        exposureInfo.setValidPolygon(None)
        exposureInfo.setPsf(None)
        exposureInfo.setWcs(None)
        exposureInfo.setPhotoCalib(None)
        exposureInfo.setCoaddInputs(None)
        exposureInfo.setVisitInfo(None)
        exposureInfo.setApCorrMap(None)
        for key in self.extras:
            exposureInfo.setComponent(key, None)

    def testSetExposureInfo(self):
        exposureInfo = afwImage.ExposureInfo()
        exposureInfo.setWcs(self.wcs)
        exposureInfo.setDetector(self.detector)
        gFilterLabel = afwImage.FilterLabel(band="g")
        exposureInfo.setFilter(gFilterLabel)
        exposureInfo.setId(self.id)
        maskedImage = afwImage.MaskedImageF(inFilePathSmall)
        exposure = afwImage.ExposureF(maskedImage)
        self.assertFalse(exposure.hasWcs())

        exposure.setInfo(exposureInfo)

        self.assertTrue(exposure.hasWcs())
        self.assertEqual(exposure.getWcs().getPixelOrigin(),
                         self.wcs.getPixelOrigin())
        self.assertEqual(exposure.getDetector().getName(),
                         self.detector.getName())
        self.assertEqual(exposure.getDetector().getSerial(),
                         self.detector.getSerial())
        self.assertEqual(exposure.getFilter(), gFilterLabel)

        # test properties
        self.assertEqual(exposure.detector.getName(), self.detector.getName())
        self.assertEqual(exposure.filter, gFilterLabel)
        self.assertEqual(exposure.wcs, self.wcs)

    def testVisitInfoFitsPersistence(self):
        """Test saving an exposure to FITS and reading it back in preserves (some) VisitInfo fields"""
        exposureTime = 12.3
        boresightRotAngle = 45.6 * lsst.geom.degrees
        weather = Weather(1.1, 2.2, 0.3)
        visitInfo = afwImage.VisitInfo(
            exposureTime=exposureTime,
            boresightRotAngle=boresightRotAngle,
            weather=weather,
        )
        photoCalib = afwImage.PhotoCalib(3.4, 5.6)
        exposureInfo = afwImage.ExposureInfo()
        exposureInfo.setVisitInfo(visitInfo)
        exposureInfo.setPhotoCalib(photoCalib)
        exposureInfo.setDetector(self.detector)
        gFilterLabel = afwImage.FilterLabel(band="g")
        exposureInfo.setFilter(gFilterLabel)
        maskedImage = afwImage.MaskedImageF(inFilePathSmall)
        exposure = afwImage.ExposureF(maskedImage, exposureInfo)
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            exposure.writeFits(tmpFile)
            rtExposure = afwImage.ExposureF(tmpFile)
        rtVisitInfo = rtExposure.getInfo().getVisitInfo()
        self.assertEqual(rtVisitInfo.getWeather(), weather)
        self.assertEqual(rtExposure.getPhotoCalib(), photoCalib)
        self.assertEqual(rtExposure.getFilter(), gFilterLabel)

        # Test property getters.
        self.assertEqual(rtExposure.photoCalib, photoCalib)
        self.assertEqual(rtExposure.filter, gFilterLabel)
        # NOTE: we can't test visitInfo equality, because most fields are NaN.
        self.assertIsNotNone(rtExposure.visitInfo)

    def testSetMembers(self):
        """
        Test that the MaskedImage and the WCS of an Exposure can be set.
        """
        exposure = afwImage.ExposureF()

        maskedImage = afwImage.MaskedImageF(inFilePathSmall)
        exposure.setMaskedImage(maskedImage)
        exposure.setWcs(self.wcs)
        exposure.setDetector(self.detector)
        exposure.setFilter(afwImage.FilterLabel(band="g"))

        self.assertEqual(exposure.getDetector().getName(),
                         self.detector.getName())
        self.assertEqual(exposure.getDetector().getSerial(),
                         self.detector.getSerial())
        self.assertEqual(exposure.getFilter().bandLabel, "g")
        self.assertEqual(exposure.getWcs(), self.wcs)

        # The PhotoCalib tests are in test_photoCalib.py;
        # here we just check that it's gettable and settable.
        self.assertIsNone(exposure.getPhotoCalib())

        photoCalib = afwImage.PhotoCalib(511.1, 44.4)
        exposure.setPhotoCalib(photoCalib)
        self.assertEqual(exposure.getPhotoCalib(), photoCalib)

        # Psfs next
        self.assertFalse(exposure.hasPsf())
        exposure.setPsf(self.psf)
        self.assertTrue(exposure.hasPsf())

        exposure.setPsf(DummyPsf(1.0))  # we can reset the Psf

        # extras next
        info = exposure.getInfo()
        for key, value in self.extras.items():
            self.assertFalse(info.hasComponent(key))
            self.assertIsNone(info.getComponent(key))
            info.setComponent(key, value)
            self.assertTrue(info.hasComponent(key))
            self.assertEqual(info.getComponent(key), value)
            info.removeComponent(key)
            self.assertFalse(info.hasComponent(key))

        # Test that we can set the MaskedImage and WCS of an Exposure
        # that already has both
        self.exposureMiWcs.setMaskedImage(maskedImage)
        exposure.setWcs(self.wcs)

    def testHasWcs(self):
        """
        Test if an Exposure has a WCS or not.
        """
        self.assertFalse(self.exposureBlank.hasWcs())

        self.assertFalse(self.exposureMiOnly.hasWcs())
        self.assertTrue(self.exposureMiWcs.hasWcs())
        self.assertTrue(self.exposureCrWcs.hasWcs())
        self.assertFalse(self.exposureCrOnly.hasWcs())

    def testGetSubExposure(self):
        """
        Test that a subExposure of the original Exposure can be obtained.

        The MaskedImage class should throw a
        lsst::pex::exceptions::InvalidParameter if the requested
        subRegion is not fully contained within the original
        MaskedImage.

        """
        #
        # This subExposure is valid
        #
        subBBox = lsst.geom.Box2I(lsst.geom.Point2I(40, 50),
                                  lsst.geom.Extent2I(10, 10))
        subExposure = self.exposureCrWcs.Factory(
            self.exposureCrWcs, subBBox, afwImage.LOCAL)

        self.checkWcs(self.exposureCrWcs, subExposure)

        # this subRegion is not valid and should trigger an exception
        # from the MaskedImage class and should trigger an exception
        # from the WCS class for the MaskedImage 871034p_1_MI.

        subRegion3 = lsst.geom.Box2I(lsst.geom.Point2I(100, 100),
                                     lsst.geom.Extent2I(10, 10))

        def getSubRegion():
            self.exposureCrWcs.Factory(
                self.exposureCrWcs, subRegion3, afwImage.LOCAL)

        self.assertRaises(pexExcept.LengthError, getSubRegion)

        # this subRegion is not valid and should trigger an exception
        # from the MaskedImage class only for the MaskedImage small_MI.
        # small_MI (cols, rows) = (256, 256)

        subRegion4 = lsst.geom.Box2I(lsst.geom.Point2I(250, 250),
                                     lsst.geom.Extent2I(10, 10))

        def getSubRegion():
            self.exposureCrWcs.Factory(
                self.exposureCrWcs, subRegion4, afwImage.LOCAL)

        self.assertRaises(pexExcept.LengthError, getSubRegion)

        # check the sub- and parent- exposures are using the same Wcs
        # transformation
        subBBox = lsst.geom.Box2I(lsst.geom.Point2I(40, 50),
                                  lsst.geom.Extent2I(10, 10))
        subExposure = self.exposureCrWcs.Factory(
            self.exposureCrWcs, subBBox, afwImage.LOCAL)
        parentSkyPos = self.exposureCrWcs.getWcs().pixelToSky(0, 0)

        subExpSkyPos = subExposure.getWcs().pixelToSky(0, 0)

        self.assertSpherePointsAlmostEqual(parentSkyPos, subExpSkyPos, msg="Wcs in sub image has changed")

    def testReadWriteFits(self):
        """Test readFits and writeFits.
        """
        # This should pass without an exception
        mainExposure = afwImage.ExposureF(inFilePathSmall)
        mainExposure.info.setId(self.id)
        mainExposure.setDetector(self.detector)

        subBBox = lsst.geom.Box2I(lsst.geom.Point2I(10, 10),
                                  lsst.geom.Extent2I(40, 50))
        subExposure = mainExposure.Factory(
            mainExposure, subBBox, afwImage.LOCAL)
        self.checkWcs(mainExposure, subExposure)
        det = subExposure.getDetector()
        self.assertTrue(det)

        subExposure = afwImage.ExposureF(
            inFilePathSmall, subBBox, afwImage.LOCAL)

        self.checkWcs(mainExposure, subExposure)

        # This should throw an exception
        def getExposure():
            afwImage.ExposureF(inFilePathSmallImage)

        self.assertRaises(FitsError, getExposure)

        mainExposure.setPsf(self.psf)

        # Make sure we can write without an exception
        photoCalib = afwImage.PhotoCalib(1e-10, 1e-12)
        mainExposure.setPhotoCalib(photoCalib)

        mainInfo = mainExposure.getInfo()
        for key, value in self.extras.items():
            mainInfo.setComponent(key, value)

        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            mainExposure.writeFits(tmpFile)

            readExposure = type(mainExposure)(tmpFile)

            #
            # Check the round-tripping
            #
            self.assertIsNotNone(mainExposure.getFilter())
            self.assertEqual(mainExposure.getFilter(),
                             readExposure.getFilter())

            self.assertEqual(photoCalib, readExposure.getPhotoCalib())

            readInfo = readExposure.getInfo()
            self.assertEqual(mainExposure.info.getId(), readInfo.id)
            for key, value in self.extras.items():
                self.assertEqual(value, readInfo.getComponent(key))

            psf = readExposure.getPsf()
            self.assertIsNotNone(psf)
            self.assertEqual(psf, self.psf)
            # check psf property getter
            self.assertEqual(readExposure.psf, self.psf)

    def checkWcs(self, parentExposure, subExposure):
        """Compare WCS at corner points of a sub-exposure and its parent exposure
           By using the function indexToPosition, we should be able to convert the indices
           (of the four corners (of the sub-exposure)) to positions and use the wcs
           to get the same sky coordinates for each.
        """
        subMI = subExposure.getMaskedImage()
        subDim = subMI.getDimensions()

        # Note: pixel positions must be computed relative to XY0 when working
        # with WCS
        mainWcs = parentExposure.getWcs()
        subWcs = subExposure.getWcs()

        for xSubInd in (0, subDim.getX()-1):
            for ySubInd in (0, subDim.getY()-1):
                self.assertSpherePointsAlmostEqual(
                    mainWcs.pixelToSky(
                        afwImage.indexToPosition(xSubInd),
                        afwImage.indexToPosition(ySubInd),
                    ),
                    subWcs.pixelToSky(
                        afwImage.indexToPosition(xSubInd),
                        afwImage.indexToPosition(ySubInd),
                    ))

    def cmpExposure(self, e1, e2):
        self.assertEqual(e1.getDetector().getName(),
                         e2.getDetector().getName())
        self.assertEqual(e1.getDetector().getSerial(),
                         e2.getDetector().getSerial())
        self.assertEqual(e1.getFilter(), e2.getFilter())
        xy = lsst.geom.Point2D(0, 0)
        self.assertEqual(e1.getWcs().pixelToSky(xy)[0],
                         e2.getWcs().pixelToSky(xy)[0])
        self.assertEqual(e1.getPhotoCalib(), e2.getPhotoCalib())
        # check PSF identity
        if not e1.getPsf():
            self.assertFalse(e2.getPsf())
        else:
            self.assertEqual(e1.getPsf(), e2.getPsf())
        # Check extra components
        i1 = e1.getInfo()
        i2 = e2.getInfo()
        for key in self.extras:
            self.assertEqual(i1.hasComponent(key), i2.hasComponent(key))
            if i1.hasComponent(key):
                self.assertEqual(i1.getComponent(key), i2.getComponent(key))

    def testCopyExposure(self):
        """Copy an Exposure (maybe changing type)"""

        exposureU = afwImage.ExposureU(inFilePathSmall, allowUnsafe=True)
        exposureU.setWcs(self.wcs)
        exposureU.setDetector(self.detector)
        exposureU.setFilter(afwImage.FilterLabel(band="g"))
        exposureU.setPsf(DummyPsf(4.0))
        infoU = exposureU.getInfo()
        for key, value in self.extras.items():
            infoU.setComponent(key, value)

        exposureF = exposureU.convertF()
        self.cmpExposure(exposureF, exposureU)

        nexp = exposureF.Factory(exposureF, False)
        self.cmpExposure(exposureF, nexp)

        # Ensure that the copy was deep.
        # (actually this test is invalid since getDetector() returns a shared_ptr)
        # cen0 = exposureU.getDetector().getCenterPixel()
        # x0,y0 = cen0
        # det = exposureF.getDetector()
        # det.setCenterPixel(lsst.geom.Point2D(999.0, 437.8))
        # self.assertEqual(exposureU.getDetector().getCenterPixel()[0], x0)
        # self.assertEqual(exposureU.getDetector().getCenterPixel()[1], y0)

    def testDeepCopyData(self):
        """Make sure a deep copy of an Exposure has its own data (ticket #2625)
        """
        exp = afwImage.ExposureF(6, 7)
        mi = exp.getMaskedImage()
        mi.getImage().set(100)
        mi.getMask().set(5)
        mi.getVariance().set(200)

        expCopy = exp.clone()
        miCopy = expCopy.getMaskedImage()
        miCopy.getImage().set(-50)
        miCopy.getMask().set(2)
        miCopy.getVariance().set(175)

        self.assertFloatsAlmostEqual(miCopy.getImage().getArray(), -50)
        self.assertTrue(np.all(miCopy.getMask().getArray() == 2))
        self.assertFloatsAlmostEqual(miCopy.getVariance().getArray(), 175)

        self.assertFloatsAlmostEqual(mi.getImage().getArray(), 100)
        self.assertTrue(np.all(mi.getMask().getArray() == 5))
        self.assertFloatsAlmostEqual(mi.getVariance().getArray(), 200)

    def testDeepCopySubData(self):
        """Make sure a deep copy of a subregion of an Exposure has its own data (ticket #2625)
        """
        exp = afwImage.ExposureF(6, 7)
        mi = exp.getMaskedImage()
        mi.getImage().set(100)
        mi.getMask().set(5)
        mi.getVariance().set(200)

        bbox = lsst.geom.Box2I(lsst.geom.Point2I(1, 0), lsst.geom.Extent2I(5, 4))
        expCopy = exp.Factory(exp, bbox, afwImage.PARENT, True)
        miCopy = expCopy.getMaskedImage()
        miCopy.getImage().set(-50)
        miCopy.getMask().set(2)
        miCopy.getVariance().set(175)

        self.assertFloatsAlmostEqual(miCopy.getImage().getArray(), -50)
        self.assertTrue(np.all(miCopy.getMask().getArray() == 2))
        self.assertFloatsAlmostEqual(miCopy.getVariance().getArray(), 175)

        self.assertFloatsAlmostEqual(mi.getImage().getArray(), 100)
        self.assertTrue(np.all(mi.getMask().getArray() == 5))
        self.assertFloatsAlmostEqual(mi.getVariance().getArray(), 200)

    def testDeepCopyMetadata(self):
        """Make sure a deep copy of an Exposure has a deep copy of metadata (ticket #2568)
        """
        exp = afwImage.ExposureF(10, 10)
        expMeta = exp.getMetadata()
        expMeta.set("foo", 5)
        expCopy = exp.clone()
        expCopyMeta = expCopy.getMetadata()
        expCopyMeta.set("foo", 6)
        self.assertEqual(expCopyMeta.getScalar("foo"), 6)
        # this will fail if the bug is present
        self.assertEqual(expMeta.getScalar("foo"), 5)

    def testDeepCopySubMetadata(self):
        """Make sure a deep copy of a subregion of an Exposure has a deep copy of metadata (ticket #2568)
        """
        exp = afwImage.ExposureF(10, 10)
        expMeta = exp.getMetadata()
        expMeta.set("foo", 5)
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(1, 0), lsst.geom.Extent2I(5, 5))
        expCopy = exp.Factory(exp, bbox, afwImage.PARENT, True)
        expCopyMeta = expCopy.getMetadata()
        expCopyMeta.set("foo", 6)
        self.assertEqual(expCopyMeta.getScalar("foo"), 6)
        # this will fail if the bug is present
        self.assertEqual(expMeta.getScalar("foo"), 5)

    def testMakeExposureLeaks(self):
        """Test for memory leaks in makeExposure (the test is in lsst.utils.tests.MemoryTestCase)"""
        afwImage.makeMaskedImage(afwImage.ImageU(lsst.geom.Extent2I(10, 20)))
        afwImage.makeExposure(afwImage.makeMaskedImage(
            afwImage.ImageU(lsst.geom.Extent2I(10, 20))))

    def testImageSlices(self):
        """Test image slicing, which generate sub-images using Box2I under the covers"""
        exp = afwImage.ExposureF(10, 20)
        mi = exp.getMaskedImage()
        mi.image[9, 19] = 10
        # N.b. Exposures don't support setting/getting the pixels so can't
        # replicate e.g. Image's slice tests
        sexp = exp[1:4, 6:10]
        self.assertEqual(sexp.getDimensions(), lsst.geom.ExtentI(3, 4))
        sexp = exp[:, -3:, afwImage.LOCAL]
        self.assertEqual(sexp.getDimensions(),
                         lsst.geom.ExtentI(exp.getWidth(), 3))
        self.assertEqual(sexp.maskedImage[-1, -1, afwImage.LOCAL],
                         exp.maskedImage[-1, -1, afwImage.LOCAL])

    def testConversionToScalar(self):
        """Test that even 1-pixel Exposures can't be converted to scalars"""
        im = afwImage.ExposureF(10, 20)

        # only single pixel images may be converted
        self.assertRaises(TypeError, float, im)
        # actually, can't convert (img, msk, var) to scalar
        self.assertRaises(TypeError, float, im[0, 0])

    def testReadMetadata(self):
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            self.exposureCrWcs.getMetadata().set("FRAZZLE", True)
            # This will write the main metadata (inc. FRAZZLE) to the primary HDU, and the
            # WCS to subsequent HDUs, along with INHERIT=T.
            self.exposureCrWcs.writeFits(tmpFile)
            # This should read the first non-empty HDU (i.e. it skips the primary), but
            # goes back and reads it if it finds INHERIT=T.  That should let us read
            # frazzle and the Wcs from the PropertySet returned by
            # testReadMetadata.
            md = readMetadata(tmpFile)
            wcs = afwGeom.makeSkyWcs(md, False)
            self.assertPairsAlmostEqual(wcs.getPixelOrigin(), self.wcs.getPixelOrigin())
            self.assertSpherePointsAlmostEqual(wcs.getSkyOrigin(), self.wcs.getSkyOrigin())
            assert_allclose(wcs.getCdMatrix(), self.wcs.getCdMatrix(), atol=1e-10)
            frazzle = md.getScalar("FRAZZLE")
            self.assertTrue(frazzle)

    def testArchiveKeys(self):
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            exposure1 = afwImage.ExposureF(100, 100, self.wcs)
            exposure1.setPsf(self.psf)
            exposure1.writeFits(tmpFile)
            exposure2 = afwImage.ExposureF(tmpFile)
            self.assertFalse(exposure2.getMetadata().exists("AR_ID"))
            self.assertFalse(exposure2.getMetadata().exists("PSF_ID"))
            self.assertFalse(exposure2.getMetadata().exists("WCS_ID"))

    def testTicket2861(self):
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            exposure1 = afwImage.ExposureF(100, 100, self.wcs)
            exposure1.setPsf(self.psf)
            schema = afwTable.ExposureTable.makeMinimalSchema()
            coaddInputs = afwImage.CoaddInputs(schema, schema)
            exposure1.getInfo().setCoaddInputs(coaddInputs)
            exposure2 = afwImage.ExposureF(exposure1, True)
            self.assertIsNotNone(exposure2.getInfo().getCoaddInputs())
            exposure2.writeFits(tmpFile)
            exposure3 = afwImage.ExposureF(tmpFile)
            self.assertIsNotNone(exposure3.getInfo().getCoaddInputs())

    def testGetCutoutSky(self):
        """Test we can get cutouts in sky coordinates, so long as there is a
        valid WCS.
        """
        wcs = self.smallExposure.getWcs()

        dimensions = [lsst.geom.Extent2I(100, 50), lsst.geom.Extent2I(15, 15), lsst.geom.Extent2I(0, 10),
                      lsst.geom.Extent2I(25, 30), lsst.geom.Extent2I(15, -5),
                      2*self.smallExposure.getDimensions()]
        locations = [("center", self._getExposureCenter(self.smallExposure)),
                     ("edge", wcs.pixelToSky(lsst.geom.Point2D(0, 0))),
                     ("rounding test", wcs.pixelToSky(lsst.geom.Point2D(0.2, 0.7))),
                     ("just inside", wcs.pixelToSky(lsst.geom.Point2D(-0.5 + 1e-4, -0.5 + 1e-4))),
                     ("just outside", wcs.pixelToSky(lsst.geom.Point2D(-0.5 - 1e-4, -0.5 - 1e-4))),
                     ("outside", wcs.pixelToSky(lsst.geom.Point2D(-1000, -1000)))]
        for cutoutSize in dimensions:
            for label, cutoutCenter in locations:
                msg = 'Cutout size = %s, location = %s' % (cutoutSize, label)
                if "outside" not in label and all(cutoutSize.gt(0)):
                    cutout = self.smallExposure.getCutout(cutoutCenter, cutoutSize)
                    centerInPixels = wcs.skyToPixel(cutoutCenter)
                    precision = (1 + 1e-4)*np.sqrt(0.5)*wcs.getPixelScale(centerInPixels)
                    self._checkCutoutProperties(cutout, cutoutSize, cutoutCenter, precision, msg)
                    self._checkCutoutPixels(
                        cutout,
                        self._getValidCorners(self.smallExposure.getBBox(), cutout.getBBox()),
                        msg)

                    # Need a valid WCS
                    with self.assertRaises(pexExcept.LogicError, msg=msg):
                        self.exposureMiOnly.getCutout(cutoutCenter, cutoutSize)
                else:
                    with self.assertRaises(pexExcept.InvalidParameterError, msg=msg):
                        self.smallExposure.getCutout(cutoutCenter, cutoutSize)

    def testGetCutoutPixel(self):
        """Test that we can get cutouts in pixel coordinates, even if the
        extent is off the edge of the image, even if there is no WCS.
        """
        dimensions = [lsst.geom.Extent2I(100, 50), lsst.geom.Extent2I(15, 15), lsst.geom.Extent2I(0, 10),
                      lsst.geom.Extent2I(25, 30), lsst.geom.Extent2I(15, -5),
                      2*self.exposureMiOnly.getDimensions()]
        locations = [("center", lsst.geom.Box2D(self.exposureMiOnly.getBBox()).getCenter()),
                     ("edge", lsst.geom.Point2D(0, 0)),
                     ("rounding test", lsst.geom.Point2D(0.2, 0.7)),
                     ("just inside", lsst.geom.Point2D(-0.5 + 1e-4, -0.5 + 1e-4)),
                     # These two should raise; center must be within image box.
                     ("just outside", lsst.geom.Point2D(-0.5 - 1e-4, -0.5 - 1e-4)),
                     ("outside", lsst.geom.Point2D(-1000, -1000))]
        for cutoutSize in dimensions:
            for label, cutoutCenter in locations:
                msg = 'Cutout size = %s, location = %s' % (cutoutSize, label)
                if "outside" not in label and all(cutoutSize.gt(0)):
                    cutout = self.exposureMiOnly.getCutout(cutoutCenter, cutoutSize)
                    self._checkCutoutPixels(
                        cutout,
                        self._getValidCorners(self.exposureMiOnly.getBBox(), cutout.getBBox()),
                        msg)

                    # Same result even if there is a wcs.
                    cutoutWithWcs = self.smallExposure.getCutout(cutoutCenter, cutoutSize)
                    self.assertMaskedImagesEqual(cutout.maskedImage, cutoutWithWcs.maskedImage)

                    # Getting a cutout with a bbox should produce the same result.
                    box = lsst.geom.Box2I.makeCenteredBox(cutoutCenter, lsst.geom.Extent2I(cutoutSize))
                    cutoutBox2I = self.exposureMiOnly.getCutout(box)
                    self.assertMaskedImagesEqual(cutout.maskedImage, cutoutBox2I.maskedImage)
                else:
                    with self.assertRaises(pexExcept.InvalidParameterError, msg=msg):
                        self.exposureMiOnly.getCutout(cutoutCenter, cutoutSize)

    def testGetConvexPolygon(self):
        """Test the convex polygon."""
        # Check that we do not have a convex polygon for the plain exposure.
        self.assertIsNone(self.exposureMiOnly.convex_polygon)

        # Check that all the points in the padded bounding box are in the polygon
        bbox = self.exposureMiWcs.getBBox()
        # Grow by the default padding.
        bbox.grow(10)
        x, y = np.meshgrid(np.arange(bbox.getBeginX(), bbox.getEndX(), dtype=np.float64),
                           np.arange(bbox.getBeginY(), bbox.getEndY(), dtype=np.float64))
        wcs = self.exposureMiWcs.wcs
        ra, dec = wcs.pixelToSkyArray(x.ravel(),
                                      y.ravel())

        poly = self.exposureMiWcs.convex_polygon
        contains = poly.contains(ra, dec)
        np.testing.assert_array_equal(contains, np.ones(len(contains), dtype=bool))

        # Check that points one pixel outside of the bounding box are not in the polygon
        bbox.grow(1)

        ra, dec = wcs.pixelToSkyArray(
            np.linspace(bbox.getBeginX(), bbox.getEndX(), 100),
            np.full(100, bbox.getBeginY()))
        contains = poly.contains(ra, dec)
        np.testing.assert_array_equal(contains, np.zeros(len(contains), dtype=bool))

        ra, dec = wcs.pixelToSkyArray(
            np.linspace(bbox.getBeginX(), bbox.getEndX(), 100),
            np.full(100, bbox.getEndY()))
        contains = poly.contains(ra, dec)
        np.testing.assert_array_equal(contains, np.zeros(len(contains), dtype=bool))

        ra, dec = wcs.pixelToSkyArray(
            np.full(100, bbox.getBeginX()),
            np.linspace(bbox.getBeginY(), bbox.getEndY(), 100))
        contains = poly.contains(ra, dec)
        np.testing.assert_array_equal(contains, np.zeros(len(contains), dtype=bool))

        ra, dec = wcs.pixelToSkyArray(
            np.full(100, bbox.getEndX()),
            np.linspace(bbox.getBeginY(), bbox.getEndY(), 100))
        contains = poly.contains(ra, dec)
        np.testing.assert_array_equal(contains, np.zeros(len(contains), dtype=bool))

    def testContainsSkyCoords(self):
        """Test the sky coord containment code."""
        self.assertRaisesRegex(ValueError,
                               "Exposure does not have a valid WCS",
                               self.exposureMiOnly.containsSkyCoords,
                               0.0,
                               0.0)

        # Check that all the points within the bounding box are contained
        bbox = self.exposureMiWcs.getBBox()
        x, y = np.meshgrid(np.arange(bbox.getBeginX() + 1, bbox.getEndX() - 1),
                           np.arange(bbox.getBeginY() + 1, bbox.getEndY() - 1))
        wcs = self.exposureMiWcs.wcs
        ra, dec = wcs.pixelToSkyArray(x.ravel().astype(np.float64),
                                      y.ravel().astype(np.float64))

        contains = self.exposureMiWcs.containsSkyCoords(ra*units.radian,
                                                        dec*units.radian)
        np.testing.assert_array_equal(contains, np.ones(len(contains), dtype=bool))

        # Same test, everything in degrees.
        ra, dec = wcs.pixelToSkyArray(x.ravel().astype(np.float64),
                                      y.ravel().astype(np.float64),
                                      degrees=True)

        contains = self.exposureMiWcs.containsSkyCoords(ra*units.degree,
                                                        dec*units.degree)
        np.testing.assert_array_equal(contains, np.ones(len(contains), dtype=bool))

        # Prepend and append some positions out of the box.
        ra = np.concatenate(([300.0], ra, [180.]))
        dec = np.concatenate(([50.0], dec, [50.0]))

        # Bad NaN handling appears as a warning, not an error
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            contains = self.exposureMiWcs.containsSkyCoords(ra*units.degree,
                                                            dec*units.degree)
        compare = np.ones(len(contains), dtype=bool)
        compare[0] = False
        compare[-1] = False
        np.testing.assert_array_equal(contains, compare)

    def _checkCutoutProperties(self, cutout, size, center, precision, msg):
        """Test whether a cutout has the desired size and position.

        Parameters
        ----------
        cutout : `lsst.afw.image.Exposure`
            The cutout to test.
        size : `lsst.geom.Extent2I`
            The expected dimensions of ``cutout``.
        center : `lsst.geom.SpherePoint`
            The expected center of ``cutout``.
        precision : `lsst.geom.Angle`
            The precision to which ``center`` must match.
        msg : `str`
            An error message suffix describing test parameters.
        """
        newCenter = self._getExposureCenter(cutout)
        self.assertIsNotNone(cutout, msg=msg)
        self.assertSpherePointsAlmostEqual(newCenter, center, maxSep=precision, msg=msg)
        self.assertEqual(cutout.getWidth(), size[0], msg=msg)
        self.assertEqual(cutout.getHeight(), size[1], msg=msg)

    def _checkCutoutPixels(self, cutout, validCorners, msg):
        """Test whether a cutout has valid/empty pixels where expected.

        Parameters
        ----------
        cutout : `lsst.afw.image.Exposure`
            The cutout to test.
        validCorners : iterable of `lsst.geom.Point2I`
            The corners of ``cutout`` that should be drawn from the original image.
        msg : `str`
            An error message suffix describing test parameters.
        """
        mask = cutout.getMaskedImage().getMask()
        edgeMask = mask.getPlaneBitMask("NO_DATA")

        for corner in cutout.getBBox().getCorners():
            maskBitsSet = mask[corner] & edgeMask
            if corner in validCorners:
                self.assertEqual(maskBitsSet, 0, msg=msg)
            else:
                self.assertEqual(maskBitsSet, edgeMask, msg=msg)

    def _getExposureCenter(self, exposure):
        """Return the sky coordinates of an Exposure's center.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            The image whose center is desired.

        Returns
        -------
        center : `lsst.geom.SpherePoint`
            The position at the center of ``exposure``.
        """
        return exposure.getWcs().pixelToSky(lsst.geom.Box2D(exposure.getBBox()).getCenter())

    def _getValidCorners(self, imageBox, cutoutBox):
        """Return the corners of a cutout that are constrained by the original image.

        Parameters
        ----------
        imageBox: `lsst.geom.Extent2I`
            The bounding box of the original image.
        cutoutBox : `lsst.geom.Box2I`
            The bounding box of the cutout.

        Returns
        -------
        corners : iterable of `lsst.geom.Point2I`
            The corners that are drawn from the original image.
        """
        return [corner for corner in cutoutBox.getCorners() if corner in imageBox]


class ExposureInfoTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        super().setUp()

        self.wcs = afwGeom.makeSkyWcs(lsst.geom.Point2D(0.0, 0.0),
                                      lsst.geom.SpherePoint(2.0, 34.0, lsst.geom.degrees),
                                      np.identity(2),
                                      )
        self.photoCalib = afwImage.PhotoCalib(1.5)
        self.psf = DummyPsf(2.0)
        self.detector = DetectorWrapper().detector
        self.summaryStats = afwImage.ExposureSummaryStats(ra=100.0)
        self.polygon = afwGeom.Polygon(lsst.geom.Box2D(lsst.geom.Point2D(0.0, 0.0),
                                                       lsst.geom.Point2D(25.0, 20.0)))
        self.coaddInputs = afwImage.CoaddInputs()
        self.apCorrMap = afwImage.ApCorrMap()
        self.transmissionCurve = afwImage.TransmissionCurve.makeIdentity()

        self.exposureInfo = afwImage.ExposureInfo()
        self.gFilterLabel = afwImage.FilterLabel(band="g")
        self.exposureId = 42

    def _checkAlias(self, exposureInfo, key, value, has, get):
        self.assertFalse(has())
        self.assertFalse(exposureInfo.hasComponent(key))
        self.assertIsNone(get())
        self.assertIsNone(exposureInfo.getComponent(key))

        self.exposureInfo.setComponent(key, value)
        self.assertTrue(has())
        self.assertTrue(exposureInfo.hasComponent(key))
        self.assertIsNotNone(get())
        self.assertIsNotNone(exposureInfo.getComponent(key))
        self.assertEqual(get(), value)
        self.assertEqual(exposureInfo.getComponent(key), value)

        self.exposureInfo.removeComponent(key)
        self.assertFalse(has())
        self.assertFalse(exposureInfo.hasComponent(key))
        self.assertIsNone(get())
        self.assertIsNone(exposureInfo.getComponent(key))

    def testAliases(self):
        cls = type(self.exposureInfo)
        self._checkAlias(self.exposureInfo, cls.KEY_WCS, self.wcs,
                         self.exposureInfo.hasWcs, self.exposureInfo.getWcs)
        self._checkAlias(self.exposureInfo, cls.KEY_PSF, self.psf,
                         self.exposureInfo.hasPsf, self.exposureInfo.getPsf)
        self._checkAlias(self.exposureInfo, cls.KEY_PHOTO_CALIB, self.photoCalib,
                         self.exposureInfo.hasPhotoCalib, self.exposureInfo.getPhotoCalib)
        self._checkAlias(self.exposureInfo, cls.KEY_DETECTOR, self.detector,
                         self.exposureInfo.hasDetector, self.exposureInfo.getDetector)
        self._checkAlias(self.exposureInfo, cls.KEY_VALID_POLYGON, self.polygon,
                         self.exposureInfo.hasValidPolygon, self.exposureInfo.getValidPolygon)
        self._checkAlias(self.exposureInfo, cls.KEY_COADD_INPUTS, self.coaddInputs,
                         self.exposureInfo.hasCoaddInputs, self.exposureInfo.getCoaddInputs)
        self._checkAlias(self.exposureInfo, cls.KEY_AP_CORR_MAP, self.apCorrMap,
                         self.exposureInfo.hasApCorrMap, self.exposureInfo.getApCorrMap)
        self._checkAlias(self.exposureInfo, cls.KEY_TRANSMISSION_CURVE, self.transmissionCurve,
                         self.exposureInfo.hasTransmissionCurve, self.exposureInfo.getTransmissionCurve)
        self._checkAlias(self.exposureInfo, cls.KEY_SUMMARY_STATS, self.summaryStats,
                         self.exposureInfo.hasSummaryStats, self.exposureInfo.getSummaryStats)
        self._checkAlias(self.exposureInfo, cls.KEY_FILTER, self.gFilterLabel,
                         self.exposureInfo.hasFilter, self.exposureInfo.getFilter)

    def testId(self):
        self.exposureInfo.setVisitInfo(afwImage.VisitInfo())

        self.assertFalse(self.exposureInfo.hasId())
        self.assertIsNone(self.exposureInfo.getId())
        self.assertIsNone(self.exposureInfo.id)

        self.exposureInfo.setId(self.exposureId)
        self.assertTrue(self.exposureInfo.hasId())
        self.assertIsNotNone(self.exposureInfo.getId())
        self.assertIsNotNone(self.exposureInfo.id)
        self.assertEqual(self.exposureInfo.getId(), self.exposureId)
        self.assertEqual(self.exposureInfo.id, self.exposureId)

        self.exposureInfo.id = 99899
        self.assertEqual(self.exposureInfo.getId(), 99899)

        self.exposureInfo.id = None
        self.assertFalse(self.exposureInfo.hasId())
        self.assertIsNone(self.exposureInfo.getId())
        self.assertIsNone(self.exposureInfo.id)

    def testCopy(self):
        # Test that ExposureInfos have independently settable state
        copy = afwImage.ExposureInfo(self.exposureInfo, True)
        self.assertEqual(self.exposureInfo.getWcs(), copy.getWcs())

        newWcs = afwGeom.makeSkyWcs(lsst.geom.Point2D(-23.0, 8.0),
                                    lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
                                    np.identity(2),
                                    )
        copy.setWcs(newWcs)
        self.assertEqual(copy.getWcs(), newWcs)
        self.assertNotEqual(self.exposureInfo.getWcs(), copy.getWcs())

    def testMissingProperties(self):
        # Test that invalid properties return None instead of raising
        exposureInfo = afwImage.ExposureInfo()

        self.assertIsNone(exposureInfo.id)


class ExposureNoAfwdataTestCase(lsst.utils.tests.TestCase):
    """Tests of Exposure that don't require afwdata.

    These tests use the trivial exposures written to ``afw/tests/data``.
    """
    def setUp(self):
        self.dataDir = os.path.join(os.path.split(__file__)[0], "data")

        # Check the values below against what was written by comparing with
        # the code in `afw/tests/data/makeTestExposure.py`
        nx = ny = 10
        image = afwImage.ImageF(np.arange(nx*ny, dtype='f').reshape(nx, ny))
        variance = afwImage.ImageF(np.ones((nx, ny), dtype='f'))
        mask = afwImage.MaskX(nx, ny)
        mask.array[5, 5] = 5
        self.maskedImage = afwImage.MaskedImageF(image, mask, variance)
        self.exposureId = 12345

        self.v0PhotoCalib = afwImage.makePhotoCalibFromCalibZeroPoint(1e6, 2e4)
        self.v1PhotoCalib = afwImage.PhotoCalib(1e6, 2e4)
        self.v1FilterLabel = afwImage.FilterLabel(physical="ha")
        self.v2FilterLabel = afwImage.FilterLabel(band="N656", physical="ha")

    def testReadUnversioned(self):
        """Test that we can read an unversioned (implicit verison 0) file.
        """
        filename = os.path.join(self.dataDir, "exposure-noversion.fits")
        exposure = afwImage.ExposureF.readFits(filename)

        self.assertMaskedImagesEqual(exposure.maskedImage, self.maskedImage)

        self.assertEqual(exposure.info.id, self.exposureId)
        self.assertEqual(exposure.getPhotoCalib(), self.v0PhotoCalib)
        self.assertEqual(exposure.getFilter(), self.v1FilterLabel)

    def testReadVersion0(self):
        """Test that we can read a version 0 file.
        This file should be identical to the unversioned one, except that it
        is marked as ExposureInfo version 0 in the header.
        """
        filename = os.path.join(self.dataDir, "exposure-version-0.fits")
        exposure = afwImage.ExposureF.readFits(filename)

        self.assertMaskedImagesEqual(exposure.maskedImage, self.maskedImage)

        self.assertEqual(exposure.info.id, self.exposureId)
        self.assertEqual(exposure.getPhotoCalib(), self.v0PhotoCalib)
        self.assertEqual(exposure.getFilter(), self.v1FilterLabel)

        # Check that the metadata reader parses the file correctly
        reader = afwImage.ExposureFitsReader(filename)
        self.assertEqual(reader.readExposureInfo().getPhotoCalib(), self.v0PhotoCalib)
        self.assertEqual(reader.readPhotoCalib(), self.v0PhotoCalib)

    def testReadVersion1(self):
        """Test that we can read a version 1 file.
        Version 1 replaced Calib with PhotoCalib.
        """
        filename = os.path.join(self.dataDir, "exposure-version-1.fits")
        exposure = afwImage.ExposureF.readFits(filename)

        self.assertMaskedImagesEqual(exposure.maskedImage, self.maskedImage)

        self.assertEqual(exposure.info.id, self.exposureId)
        self.assertEqual(exposure.getPhotoCalib(), self.v1PhotoCalib)
        self.assertEqual(exposure.getFilter(), self.v1FilterLabel)

        # Check that the metadata reader parses the file correctly
        reader = afwImage.ExposureFitsReader(filename)
        self.assertEqual(reader.readExposureInfo().getPhotoCalib(), self.v1PhotoCalib)
        self.assertEqual(reader.readPhotoCalib(), self.v1PhotoCalib)

    def testReadVersion2(self):
        """Test that we can read a version 2 file.
        Version 2 replaced Filter with FilterLabel.
        """
        filename = os.path.join(self.dataDir, "exposure-version-2.fits")
        exposure = afwImage.ExposureF.readFits(filename)

        self.assertMaskedImagesEqual(exposure.maskedImage, self.maskedImage)

        self.assertEqual(exposure.info.id, self.exposureId)
        self.assertEqual(exposure.getPhotoCalib(), self.v1PhotoCalib)
        self.assertEqual(exposure.getFilter(), self.v2FilterLabel)

        # Check that the metadata reader parses the file correctly
        reader = afwImage.ExposureFitsReader(filename)
        self.assertEqual(reader.readExposureInfo().getPhotoCalib(), self.v1PhotoCalib)
        self.assertEqual(reader.readPhotoCalib(), self.v1PhotoCalib)

    def testReadDottedHeaderKey(self):
        """Test that we can read a file with a dot-delimited header key."""
        original = afwImage.ExposureF.readFits(os.path.join(self.dataDir, "exposure-version-2.fits"))
        original.metadata["x.y.z"] = "three"
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            original.writeFits(tmpFile)
            roundtripped = afwImage.ExposureF(tmpFile)
        self.assertMaskedImagesEqual(original.maskedImage, roundtripped.maskedImage)

    def testExposureSummaryExtraComponents(self):
        """Test that we can read an exposure summary with extra components.
        """
        testDict = {'ra': 0.0,
                    'dec': 0.0,
                    'nonsense': 1.0}
        bytes = yaml.dump(testDict, encoding='utf-8')
        with self.assertWarns(FutureWarning):
            summaryStats = lsst.afw.image.ExposureSummaryStats._read(bytes)

        self.assertEqual(summaryStats.ra, testDict['ra'])
        self.assertEqual(summaryStats.dec, testDict['dec'])

    def testExposureSummaryForwardComponents(self):
        """Test that we can forward extra components (e.g. decl->dec).
        """
        testDict = {'ra': 10.0,
                    'decl': 10.0}
        bytes = yaml.dump(testDict, encoding='utf-8')
        # Cleanly forwarded fields must not result in a warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            summaryStats = lsst.afw.image.ExposureSummaryStats._read(bytes)

        self.assertEqual(summaryStats.ra, testDict['ra'])
        self.assertEqual(summaryStats.dec, testDict['decl'])

        # And check if there are both listed, it should use the new dec value.
        testDict = {'ra': 10.0,
                    'dec': 5.0,
                    'decl': 10.0}
        bytes = yaml.dump(testDict, encoding='utf-8')
        with self.assertWarns(FutureWarning):
            summaryStats = lsst.afw.image.ExposureSummaryStats._read(bytes)

        self.assertEqual(summaryStats.ra, testDict['ra'])
        self.assertEqual(summaryStats.dec, testDict['dec'])

    def testExposureSummarySchema(self):
        """Test that we can make a schema for an exposure summary and populate
        records with that schema.
        """
        schema = afwTable.Schema()
        afwImage.ExposureSummaryStats.update_schema(schema)
        self.maxDiff = None
        self.assertEqual(
            {field.name for field in dataclasses.fields(afwImage.ExposureSummaryStats)},
            set(schema.getNames()) | {"version"},
        )
        catalog = afwTable.BaseCatalog(schema)
        summary1 = afwImage.ExposureSummaryStats()
        for n, field in enumerate(dataclasses.fields(afwImage.ExposureSummaryStats)):
            # Set fields to deterministic, distinct, but arbitrary values.
            if field.type == "float":
                setattr(summary1, field.name, float(0.5**n))
            elif field.type == "int":
                setattr(summary1, field.name, 10*n)
            elif field.type == "list[float]":
                setattr(summary1, field.name, [n + 0.1, n + 0.2, n + 0.3, n + 0.4])
            else:
                raise TypeError(f"Unexpected type: {field.type!r}.")
        record = catalog.addNew()
        summary1.update_record(record)
        summary2 = afwImage.ExposureSummaryStats.from_record(record)
        self.assertEqual(summary1, summary2)

    def testMetadataProperty(self):
        """Test that the metadata property works as expected.
        """
        exposure = afwImage.ExposureF(3, 4)
        self.assertFalse(exposure.metadata)
        self.assertIsNotNone(exposure.metadata)
        exposure.metadata = None
        self.assertIsNone(exposure.metadata)
        metadata = PropertyList()
        metadata["one"] = 1
        exposure.metadata = metadata
        self.assertEqual(exposure.metadata["one"], 1)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
