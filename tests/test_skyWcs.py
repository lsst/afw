import itertools
import os
import sys
import unittest

import astropy.io.fits
import astropy.coordinates
import astropy.wcs
import astshim as ast
import numpy as np
from numpy.testing import assert_allclose

from lsst.daf.base import PropertyList
import lsst.geom
import lsst.afw.cameraGeom as cameraGeom
from lsst.afw.geom import (
    TransformPoint2ToPoint2, TransformPoint2ToSpherePoint, makeRadialTransform,
    SkyWcs, makeSkyWcs, makeCdMatrix, makeWcsPairTransform,
    makeFlippedWcs, makeModifiedWcs, makeTanSipWcs,
    getIntermediateWorldCoordsToSky, getPixelToIntermediateWorldCoords,
    stripWcsMetadata
)
from lsst.afw.geom import getCdMatrixFromMetadata, getSipMatrixFromMetadata, makeSimpleWcsMetadata
from lsst.afw.geom.testUtils import makeSipIwcToPixel, makeSipPixelToIwc
from lsst.afw.fits import makeLimitedFitsHeader
from lsst.afw.image import ExposureF


def addActualPixelsFrame(skyWcs, actualPixelsToPixels):
    """Add an "ACTUAL_PIXELS" frame to a SkyWcs and return the result

    Parameters
    ----------
    skyWcs : `lsst.afw.geom.SkyWcs`
        The WCS to which you wish to add an ACTUAL_PIXELS frame
    actualPixelsToPixels : `lsst.afw.geom.TransformPoint2ToPoint2`
        The transform from ACTUAL_PIXELS to PIXELS
    """
    actualPixelsToPixelsMap = actualPixelsToPixels.getMapping()
    actualPixelsFrame = ast.Frame(2, "Domain=ACTUAL_PIXELS")
    frameDict = skyWcs.getFrameDict()
    frameDict.addFrame("PIXELS", actualPixelsToPixelsMap.inverted(), actualPixelsFrame)
    frameDict.setBase("ACTUAL_PIXELS")
    frameDict.setCurrent("SKY")
    return SkyWcs(frameDict)


class SkyWcsBaseTestCase(lsst.utils.tests.TestCase):
    def checkPersistence(self, skyWcs, bbox):
        """Check persistence of a SkyWcs
        """
        className = "SkyWcs"

        # check writeString and readString
        skyWcsStr = skyWcs.writeString()
        serialVersion, serialClassName, serialRest = skyWcsStr.split(" ", 2)
        self.assertEqual(int(serialVersion), 1)
        self.assertEqual(serialClassName, className)
        badStr1 = " ".join(["2", serialClassName, serialRest])
        with self.assertRaises(lsst.pex.exceptions.TypeError):
            skyWcs.readString(badStr1)
        badClassName = "x" + serialClassName
        badStr2 = " ".join(["1", badClassName, serialRest])
        with self.assertRaises(lsst.pex.exceptions.TypeError):
            skyWcs.readString(badStr2)
        skyWcsFromStr1 = skyWcs.readString(skyWcsStr)
        self.assertEqual(skyWcs, skyWcsFromStr1)
        self.assertEqual(type(skyWcs), type(skyWcsFromStr1))
        self.assertEqual(skyWcs.getFrameDict(), skyWcsFromStr1.getFrameDict())

        pixelPoints = [
            lsst.geom.Point2D(0, 0),
            lsst.geom.Point2D(1000, 0),
            lsst.geom.Point2D(0, 1000),
            lsst.geom.Point2D(-50, -50),
        ]
        skyPoints = skyWcs.pixelToSky(pixelPoints)
        pixelPoints2 = skyWcs.skyToPixel(skyPoints)
        assert_allclose(pixelPoints, pixelPoints2, atol=1e-7)

        # check that WCS is properly saved as part of an exposure FITS file
        exposure = ExposureF(100, 100, skyWcs)
        with lsst.utils.tests.getTempFilePath(".fits") as outFile:
            exposure.writeFits(outFile)
            exposureRoundTrip = ExposureF(outFile)
        wcsFromExposure = exposureRoundTrip.getWcs()
        self.assertWcsAlmostEqualOverBBox(skyWcs, wcsFromExposure, bbox, maxDiffPix=0,
                                          maxDiffSky=0*lsst.geom.radians)

    def checkFrameDictConstructor(self, skyWcs, bbox):
        """Check that the FrameDict constructor works
        """
        frameDict = skyWcs.getFrameDict()
        wcsFromFrameDict = SkyWcs(frameDict)
        self.assertWcsAlmostEqualOverBBox(skyWcs, wcsFromFrameDict, bbox, maxDiffPix=0,
                                          maxDiffSky=0*lsst.geom.radians)

        self.checkPersistence(wcsFromFrameDict, bbox)

        # check that it is impossible to build a SkyWcs if a required frame is missing
        for domain in ("PIXELS", "IWC", "SKY"):
            badFrameDict = skyWcs.getFrameDict()
            badFrameDict.removeFrame(domain)
            with self.assertRaises(lsst.pex.exceptions.TypeError):
                SkyWcs(badFrameDict)

    def checkMakeFlippedWcs(self, skyWcs, skyAtol=1e-7*lsst.geom.arcseconds):
        """Check makeFlippedWcs on the provided WCS
        """
        # make an arbitrary bbox, but one that includes zero in one axis
        # and does not include zero in the other axis
        # the center of the bbox is used as the center of flipping
        # and the corners of the bbox are the input positions that are tested
        bbox = lsst.geom.Box2D(lsst.geom.Point2D(-100, 1000), lsst.geom.Extent2D(2000, 1501))
        # dict of (isRight, isTop): position
        minPos = bbox.getMin()
        maxPos = bbox.getMax()
        center = bbox.getCenter()
        cornerDict = {
            (False, False): minPos,
            (False, True): lsst.geom.Point2D(minPos[0], maxPos[1]),
            (True, False): lsst.geom.Point2D(maxPos[0], minPos[1]),
            (True, True): maxPos,
        }
        for flipLR, flipTB in itertools.product((False, True), (False, True)):
            flippedWcs = makeFlippedWcs(wcs=skyWcs, flipLR=flipLR, flipTB=flipTB, center=center)
            # the center is unchanged
            self.assertSpherePointsAlmostEqual(skyWcs.pixelToSky(center),
                                               flippedWcs.pixelToSky(center), maxSep=skyAtol)

            for isR, isT in itertools.product((False, True), (False, True)):
                origPos = cornerDict[(isR, isT)]
                flippedPos = cornerDict[(isR ^ flipLR, isT ^ flipTB)]
                self.assertSpherePointsAlmostEqual(skyWcs.pixelToSky(origPos),
                                                   flippedWcs.pixelToSky(flippedPos), maxSep=skyAtol)

    def assertSkyWcsAstropyWcsAlmostEqual(self, skyWcs, astropyWcs, bbox,
                                          pixAtol=1e-4, skyAtol=1e-4*lsst.geom.arcseconds,
                                          checkRoundTrip=True):
        """Assert that a SkyWcs and the corresponding astropy.wcs.WCS agree over a specified bounding box
        """
        bbox = lsst.geom.Box2D(bbox)
        center = bbox.getCenter()
        xArr = bbox.getMinX(), center[0], bbox.getMaxX()
        yArr = bbox.getMinY(), center[1], bbox.getMaxY()
        pixPosList = [lsst.geom.Point2D(x, y) for x, y in itertools.product(xArr, yArr)]

        # pixelToSky
        skyPosList = skyWcs.pixelToSky(pixPosList)
        astropySkyPosList = self.astropyPixelsToSky(astropyWcs=astropyWcs, pixPosList=pixPosList)
        self.assertSpherePointListsAlmostEqual(skyPosList, astropySkyPosList, maxSep=skyAtol)

        if not checkRoundTrip:
            return

        # astropy round trip
        astropyPixPosRoundTrip = self.astropySkyToPixels(astropyWcs=astropyWcs, skyPosList=astropySkyPosList)
        self.assertPairListsAlmostEqual(pixPosList, astropyPixPosRoundTrip, maxDiff=pixAtol)

        # SkyWcs round trip
        pixPosListRoundTrip = skyWcs.skyToPixel(skyPosList)
        self.assertPairListsAlmostEqual(pixPosList, pixPosListRoundTrip, maxDiff=pixAtol)

        # skyToPixel astropy vs SkyWcs
        astropyPixPosList2 = self.astropySkyToPixels(astropyWcs=astropyWcs, skyPosList=skyPosList)
        self.assertPairListsAlmostEqual(pixPosListRoundTrip, astropyPixPosList2, maxDiff=pixAtol)

    def astropyPixelsToSky(self, astropyWcs, pixPosList):
        """Use an astropy wcs to convert pixels to sky

        @param[in] astropyWcs  a celestial astropy.wcs.WCS with 2 axes in RA, Dec order
        @param[in] pixPosList 0-based pixel positions as lsst.geom.Point2D or similar pairs
        @returns sky coordinates as a list of lsst.geom.SpherePoint

        Converts the output to ICRS
        """
        xarr = [p[0] for p in pixPosList]
        yarr = [p[1] for p in pixPosList]
        skyCoordList = astropy.wcs.utils.pixel_to_skycoord(xp=xarr,
                                                           yp=yarr,
                                                           wcs=astropyWcs,
                                                           origin=0,
                                                           mode="all")
        icrsList = [sc.transform_to("icrs") for sc in skyCoordList]
        return [lsst.geom.SpherePoint(sc.ra.deg, sc.dec.deg, lsst.geom.degrees) for sc in icrsList]

    def astropySkyToPixels(self, astropyWcs, skyPosList):
        """Use an astropy wcs to convert pixels to sky

        @param[in] astropyWcs  a celestial astropy.wcs.WCS with 2 axes in RA, Dec order
        @param[in] skyPosList ICRS sky coordinates as a list of lsst.geom.SpherePoint
        @returns a list of lsst.geom.Point2D, 0-based pixel positions

        Converts the input from ICRS to the coordinate system of the wcs
        """
        skyCoordList = [astropy.coordinates.SkyCoord(c[0].asDegrees(),
                                                     c[1].asDegrees(),
                                                     frame="icrs",
                                                     unit="deg") for c in skyPosList]
        xyArr = [astropy.wcs.utils.skycoord_to_pixel(coords=sc,
                                                     wcs=astropyWcs,
                                                     origin=0,
                                                     mode="all") for sc in skyCoordList]
        # float is needed to avoid truncation to int
        return [lsst.geom.Point2D(float(x), float(y)) for x, y in xyArr]


class SimpleSkyWcsTestCase(SkyWcsBaseTestCase):
    """Test the simple FITS version of makeSkyWcs
    """

    def setUp(self):
        self.crpix = lsst.geom.Point2D(100, 100)
        self.crvalList = [
            lsst.geom.SpherePoint(0, 45, lsst.geom.degrees),
            lsst.geom.SpherePoint(0.00001, 45, lsst.geom.degrees),
            lsst.geom.SpherePoint(359.99999, 45, lsst.geom.degrees),
            lsst.geom.SpherePoint(30, 89.99999, lsst.geom.degrees),
            lsst.geom.SpherePoint(30, -89.99999, lsst.geom.degrees),
        ]
        self.orientationList = [
            0 * lsst.geom.degrees,
            0.00001 * lsst.geom.degrees,
            -0.00001 * lsst.geom.degrees,
            -45 * lsst.geom.degrees,
            90 * lsst.geom.degrees,
        ]
        self.scale = 1.0 * lsst.geom.arcseconds
        self.tinyPixels = 1.0e-10
        self.tinyAngle = 1.0e-10 * lsst.geom.radians
        self.bbox = lsst.geom.Box2I(lsst.geom.Point2I(-1000, -1000),
                                    lsst.geom.Extent2I(2000, 2000))  # arbitrary but reasonable

    def checkTanWcs(self, crval, orientation, flipX):
        """Construct a pure TAN SkyWcs and check that it operates as specified

        Parameters
        ----------
        crval : `lsst.geom.SpherePoint`
            Desired reference sky position.
            Must not be at either pole.
        orientation : `lsst.geom.Angle`
            Position angle of pixel +Y, measured from N through E.
            At 0 degrees, +Y is along N and +X is along E/W if flipX false/true
            At 90 degrees, +Y is along E and +X is along S/N if flipX false/true
        flipX : `bool`
            Flip x axis? See `orientation` for details.

        Returns
        -------
        wcs : `lsst.afw.geom.SkyWcs`
            The generated pure TAN SkyWcs
        """
        cdMatrix = makeCdMatrix(scale=self.scale, orientation=orientation, flipX=flipX)
        wcs = makeSkyWcs(crpix=self.crpix, crval=crval, cdMatrix=cdMatrix)
        self.checkPersistence(wcs, bbox=self.bbox)
        self.checkMakeFlippedWcs(wcs)

        self.assertTrue(wcs.isFits)
        self.assertEqual(wcs.isFlipped, bool(flipX))

        xoffAng = 0*lsst.geom.degrees if flipX else 180*lsst.geom.degrees

        pixelList = [
            lsst.geom.Point2D(self.crpix[0], self.crpix[1]),
            lsst.geom.Point2D(self.crpix[0] + 1, self.crpix[1]),
            lsst.geom.Point2D(self.crpix[0], self.crpix[1] + 1),
        ]
        skyList = wcs.pixelToSky(pixelList)

        # check pixels to sky
        predSkyList = [
            crval,
            crval.offset(xoffAng - orientation, self.scale),
            crval.offset(90*lsst.geom.degrees - orientation, self.scale),
        ]
        self.assertSpherePointListsAlmostEqual(predSkyList, skyList)
        self.assertSpherePointListsAlmostEqual(predSkyList, wcs.pixelToSky(pixelList))
        for pixel, predSky in zip(pixelList, predSkyList):
            self.assertSpherePointsAlmostEqual(predSky, wcs.pixelToSky(pixel))
            self.assertSpherePointsAlmostEqual(predSky, wcs.pixelToSky(pixel[0], pixel[1]))

        # check sky to pixels
        self.assertPairListsAlmostEqual(pixelList, wcs.skyToPixel(skyList))
        self.assertPairListsAlmostEqual(pixelList, wcs.skyToPixel(skyList))
        for pixel, sky in zip(pixelList, skyList):
            self.assertPairsAlmostEqual(pixel, wcs.skyToPixel(sky))
            # self.assertPairsAlmostEqual(pixel, wcs.skyToPixel(sky[0], sky[1]))

        # check CRVAL round trip
        self.assertSpherePointsAlmostEqual(wcs.getSkyOrigin(), crval,
                                           maxSep=self.tinyAngle)

        crpix = wcs.getPixelOrigin()
        self.assertPairsAlmostEqual(crpix, self.crpix, maxDiff=self.tinyPixels)

        self.assertFloatsAlmostEqual(wcs.getCdMatrix(), cdMatrix, atol=1e-15, rtol=1e-11)

        pixelScale = wcs.getPixelScale()
        self.assertAnglesAlmostEqual(self.scale, pixelScale, maxDiff=self.tinyAngle)

        pixelScale = wcs.getPixelScale(self.crpix)
        self.assertAnglesAlmostEqual(self.scale, pixelScale, maxDiff=self.tinyAngle)

        # check that getFitsMetadata can operate at high precision
        # and has axis order RA, Dec
        fitsMetadata = wcs.getFitsMetadata(True)
        self.assertEqual(fitsMetadata.getScalar("CTYPE1")[0:4], "RA--")
        self.assertEqual(fitsMetadata.getScalar("CTYPE2")[0:4], "DEC-")

        # Compute a WCS with the pixel origin shifted by an arbitrary amount
        # The resulting sky origin should not change
        offset = lsst.geom.Extent2D(500, -322)  # arbitrary
        shiftedWcs = wcs.copyAtShiftedPixelOrigin(offset)
        self.assertTrue(shiftedWcs.isFits)
        predShiftedPixelOrigin = self.crpix + offset
        self.assertPairsAlmostEqual(shiftedWcs.getPixelOrigin(), predShiftedPixelOrigin,
                                    maxDiff=self.tinyPixels)
        self.assertSpherePointsAlmostEqual(shiftedWcs.getSkyOrigin(), crval, maxSep=self.tinyAngle)

        shiftedPixelList = [p + offset for p in pixelList]
        shiftedSkyList = shiftedWcs.pixelToSky(shiftedPixelList)
        self.assertSpherePointListsAlmostEqual(skyList, shiftedSkyList, maxSep=self.tinyAngle)

        # Check that the shifted WCS can be round tripped as FITS metadata
        shiftedMetadata = shiftedWcs.getFitsMetadata(precise=True)
        shiftedWcsCopy = makeSkyWcs(shiftedMetadata)
        shiftedBBox = lsst.geom.Box2D(predShiftedPixelOrigin,
                                      predShiftedPixelOrigin + lsst.geom.Extent2I(2000, 2000))
        self.assertWcsAlmostEqualOverBBox(shiftedWcs, shiftedWcsCopy, shiftedBBox)

        wcsCopy = SkyWcs.readString(wcs.writeString())
        self.assertTrue(wcsCopy.isFits)

        return wcs

    def checkNonFitsWcs(self, wcs):
        """Check SkyWcs.getFitsMetadata for a WCS that cannot be represented as a FITS-WCS
        """
        # the modified WCS should not be representable as pure FITS-WCS
        self.assertFalse(wcs.isFits)
        with self.assertRaises(RuntimeError):
            wcs.getFitsMetadata(True)
        with self.assertRaises(RuntimeError):
            wcs.getFitsMetadata(False)

        # When a WCS is not valid FITS itself, we can explicitly install a FITS
        # approximation.  The 'wcsApprox' below is just arbitrary, not really
        # an approximation in this test, but since SkyWcs shouldn't be
        # responsible for the accuracy of the approximation that's fine.
        wcsApprox = makeSkyWcs(
            lsst.geom.Point2D(5, 4),
            lsst.geom.SpherePoint(20.0, 30.0, lsst.geom.degrees),
            np.identity(2),
        )
        wcsWithApproximation = wcs.copyWithFitsApproximation(wcsApprox)
        self.assertTrue(wcsWithApproximation.hasFitsApproximation())
        self.assertEqual(wcsWithApproximation.getFitsApproximation(), wcsApprox)
        # When we save a SkyWcs with a FITS approximation it round-trips.
        with lsst.utils.tests.getTempFilePath(".fits") as filePath:
            wcsWithApproximation.writeFits(filePath)
            wcsFromFits = SkyWcs.readFits(filePath)
            self.assertTrue(wcsFromFits.hasFitsApproximation())
            self.assertEqual(wcsFromFits.getFitsApproximation(), wcsApprox)
        # When we save an Exposure with a non-FITS SkyWcs that has a FITS
        # approximation, the approximation appears in the headers.
        exposure = ExposureF(lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(3, 2)))
        exposure.setWcs(wcsWithApproximation)
        with lsst.utils.tests.getTempFilePath(".fits") as filePath:
            exposure.writeFits(filePath)
            exposureFromFits = ExposureF(filePath)
            self.assertEqual(exposureFromFits.wcs, wcsWithApproximation)
            self.assertEqual(exposureFromFits.wcs.getFitsApproximation(), wcsApprox)
            with astropy.io.fits.open(filePath) as fits:
                self.assertEqual(fits[1].header["CRVAL1"], 20.0)
                self.assertEqual(fits[1].header["CRVAL2"], 30.0)

    def testTanWcs(self):
        """Check a variety of TanWcs, with crval not at a pole.
        """
        for crval, orientation, flipX in itertools.product(self.crvalList,
                                                           self.orientationList,
                                                           (False, True)):
            self.checkTanWcs(crval=crval,
                             orientation=orientation,
                             flipX=flipX,
                             )

    def testTanWcsFromFrameDict(self):
        """Test making a TAN WCS from a FrameDict
        """
        cdMatrix = makeCdMatrix(scale=self.scale)
        skyWcs = makeSkyWcs(crpix=self.crpix, crval=self.crvalList[0], cdMatrix=cdMatrix)
        self.checkFrameDictConstructor(skyWcs, bbox=self.bbox)

    def testGetFrameDict(self):
        """Test that getFrameDict returns a deep copy
        """
        cdMatrix = makeCdMatrix(scale=self.scale)
        skyWcs = makeSkyWcs(crpix=self.crpix, crval=self.crvalList[0], cdMatrix=cdMatrix)
        for domain in ("PIXELS", "IWC", "SKY"):
            frameDict = skyWcs.getFrameDict()
            frameDict.removeFrame(domain)
            self.assertFalse(frameDict.hasDomain(domain))
            self.assertTrue(skyWcs.getFrameDict().hasDomain(domain))

    def testMakeModifiedWcsNoActualPixels(self):
        """Test makeModifiedWcs on a SkyWcs that has no ACTUAL_PIXELS frame
        """
        cdMatrix = makeCdMatrix(scale=self.scale)
        originalWcs = makeSkyWcs(crpix=self.crpix, crval=self.crvalList[0], cdMatrix=cdMatrix)
        originalFrameDict = originalWcs.getFrameDict()

        # make an arbitrary but reasonable transform to insert using makeModifiedWcs
        pixelTransform = makeRadialTransform([0.0, 1.0, 0.0, 0.0011])
        # the result of the insertion should be as follows
        desiredPixelsToSky = pixelTransform.then(originalWcs.getTransform())

        pixPointList = (  # arbitrary but reasonable
            lsst.geom.Point2D(0.0, 0.0),
            lsst.geom.Point2D(1000.0, 0.0),
            lsst.geom.Point2D(0.0, 2000.0),
            lsst.geom.Point2D(-1111.0, -2222.0),
        )
        for modifyActualPixels in (False, True):
            modifiedWcs = makeModifiedWcs(pixelTransform=pixelTransform,
                                          wcs=originalWcs,
                                          modifyActualPixels=modifyActualPixels)
            modifiedFrameDict = modifiedWcs.getFrameDict()
            skyList = modifiedWcs.pixelToSky(pixPointList)

            # compare pixels to sky
            desiredSkyList = desiredPixelsToSky.applyForward(pixPointList)
            self.assertSpherePointListsAlmostEqual(skyList, desiredSkyList)

            # compare pixels to IWC
            pixelsToIwc = TransformPoint2ToPoint2(modifiedFrameDict.getMapping("PIXELS", "IWC"))
            desiredPixelsToIwc = TransformPoint2ToPoint2(
                pixelTransform.getMapping().then(originalFrameDict.getMapping("PIXELS", "IWC")))
            self.assertPairListsAlmostEqual(pixelsToIwc.applyForward(pixPointList),
                                            desiredPixelsToIwc.applyForward(pixPointList))

            self.checkNonFitsWcs(modifiedWcs)

    def testMakeModifiedWcsWithActualPixels(self):
        """Test makeModifiedWcs on a SkyWcs that has an ACTUAL_PIXELS frame
        """
        cdMatrix = makeCdMatrix(scale=self.scale)
        baseWcs = makeSkyWcs(crpix=self.crpix, crval=self.crvalList[0], cdMatrix=cdMatrix)
        # model actual pixels to pixels as an arbitrary zoom factor;
        # this is not realistic, but is fine for a unit test
        actualPixelsToPixels = TransformPoint2ToPoint2(ast.ZoomMap(2, 0.72))
        originalWcs = addActualPixelsFrame(baseWcs, actualPixelsToPixels)
        originalFrameDict = originalWcs.getFrameDict()

        # make an arbitrary but reasonable transform to insert using makeModifiedWcs
        pixelTransform = makeRadialTransform([0.0, 1.0, 0.0, 0.0011])  # arbitrary but reasonable

        pixPointList = (  # arbitrary but reasonable
            lsst.geom.Point2D(0.0, 0.0),
            lsst.geom.Point2D(1000.0, 0.0),
            lsst.geom.Point2D(0.0, 2000.0),
            lsst.geom.Point2D(-1111.0, -2222.0),
        )
        for modifyActualPixels in (True, False):
            modifiedWcs = makeModifiedWcs(pixelTransform=pixelTransform,
                                          wcs=originalWcs,
                                          modifyActualPixels=modifyActualPixels)
            modifiedFrameDict = modifiedWcs.getFrameDict()
            self.assertEqual(modifiedFrameDict.getFrame(modifiedFrameDict.BASE).domain, "ACTUAL_PIXELS")
            modifiedActualPixelsToPixels = \
                TransformPoint2ToPoint2(modifiedFrameDict.getMapping("ACTUAL_PIXELS", "PIXELS"))
            modifiedPixelsToIwc = TransformPoint2ToPoint2(modifiedFrameDict.getMapping("PIXELS", "IWC"))

            # compare pixels to sky
            skyList = modifiedWcs.pixelToSky(pixPointList)
            if modifyActualPixels:
                desiredPixelsToSky = pixelTransform.then(originalWcs.getTransform())
            else:
                originalPixelsToSky = \
                    TransformPoint2ToSpherePoint(originalFrameDict.getMapping("PIXELS", "SKY"))
                desiredPixelsToSky = actualPixelsToPixels.then(pixelTransform).then(originalPixelsToSky)
            desiredSkyList = desiredPixelsToSky.applyForward(pixPointList)
            self.assertSpherePointListsAlmostEqual(skyList, desiredSkyList)

            # compare ACTUAL_PIXELS to PIXELS and PIXELS to IWC
            if modifyActualPixels:
                # check that ACTUAL_PIXELS to PIXELS has been modified as expected
                desiredActualPixelsToPixels = pixelTransform.then(actualPixelsToPixels)
                self.assertPairListsAlmostEqual(modifiedActualPixelsToPixels.applyForward(pixPointList),
                                                desiredActualPixelsToPixels.applyForward(pixPointList))

                # check that PIXELS to IWC is unchanged
                originalPixelsToIwc = TransformPoint2ToPoint2(originalFrameDict.getMapping("PIXELS", "IWC"))
                self.assertPairListsAlmostEqual(modifiedPixelsToIwc.applyForward(pixPointList),
                                                originalPixelsToIwc.applyForward(pixPointList))

            else:
                # check that ACTUAL_PIXELS to PIXELS is unchanged
                self.assertPairListsAlmostEqual(actualPixelsToPixels.applyForward(pixPointList),
                                                actualPixelsToPixels.applyForward(pixPointList))

                # check that PIXELS to IWC has been modified as expected
                desiredPixelsToIwc = TransformPoint2ToPoint2(
                    pixelTransform.getMapping().then(originalFrameDict.getMapping("PIXELS", "IWC")))
                self.assertPairListsAlmostEqual(modifiedPixelsToIwc.applyForward(pixPointList),
                                                desiredPixelsToIwc.applyForward(pixPointList))

            self.checkNonFitsWcs(modifiedWcs)

    def testMakeSkyWcsFromPixelsToFieldAngle(self):
        """Test makeSkyWcs from a pixelsToFieldAngle transform
        """
        pixelSizeMm = 25e-3
        # place the detector in several positions at several orientations
        # use fewer CRVAL and orientations to speed up the test
        for fpPosition, yaw, addOpticalDistortion, crval, pixelOrientation, \
            flipX, projection in itertools.product(
                (lsst.geom.Point3D(0, 0, 0), lsst.geom.Point3D(-100, 500, 1.5)),
                (0*lsst.geom.degrees, 71*lsst.geom.degrees), (False, True),
                self.crvalList[0:2], self.orientationList[0:2], (False, True), ("TAN", "STG")):
            with self.subTest(fpPosition=repr(fpPosition), yaw=repr(yaw),
                              addOpticalDistortion=repr(addOpticalDistortion),
                              crval=repr(crval), orientation=repr(pixelOrientation)):
                pixelsToFocalPlane = cameraGeom.Orientation(
                    fpPosition=fpPosition,
                    yaw=yaw,
                ).makePixelFpTransform(lsst.geom.Extent2D(pixelSizeMm, pixelSizeMm))
                # Compute crpix before adding optical distortion,
                # since it is not affected by such distortion
                crpix = pixelsToFocalPlane.applyInverse(lsst.geom.Point2D(0, 0))
                radiansPerMm = self.scale.asRadians() / pixelSizeMm
                focalPlaneToFieldAngle = lsst.afw.geom.makeTransform(
                    lsst.geom.AffineTransform(lsst.geom.LinearTransform.makeScaling(radiansPerMm)))
                pixelsToFieldAngle = pixelsToFocalPlane.then(focalPlaneToFieldAngle)

                cdMatrix = makeCdMatrix(scale=self.scale, orientation=pixelOrientation, flipX=flipX)
                wcs1 = makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cdMatrix, projection=projection)

                if addOpticalDistortion:
                    # Model optical distortion as a pixel transform,
                    # so it can be added to the WCS created from crpix,
                    # cdMatrix, etc. using makeModifiedWcs
                    pixelTransform = makeRadialTransform([0.0, 1.0, 0.0, 0.0011])
                    pixelsToFieldAngle = pixelTransform.then(pixelsToFieldAngle)
                    wcs1 = makeModifiedWcs(pixelTransform=pixelTransform, wcs=wcs1, modifyActualPixels=False)

                # orientation is with respect to detector x, y
                # but this flavor of makeSkyWcs needs it with respect to focal plane x, y
                focalPlaneOrientation = pixelOrientation + (yaw if flipX else -yaw)
                wcs2 = makeSkyWcs(pixelsToFieldAngle=pixelsToFieldAngle,
                                  orientation=focalPlaneOrientation,
                                  flipX=flipX,
                                  boresight=crval,
                                  projection=projection)
                self.assertWcsAlmostEqualOverBBox(wcs1, wcs2, self.bbox)

    @unittest.skipIf(sys.version_info[0] < 3, "astropy.wcs rejects the header on py2")
    def testAgainstAstropyWcs(self):
        bbox = lsst.geom.Box2D(lsst.geom.Point2D(-1000, -1000), lsst.geom.Extent2D(2000, 2000))
        for crval, orientation, flipX, projection in itertools.product(self.crvalList,
                                                                       self.orientationList,
                                                                       (False, True),
                                                                       ("TAN", "STG", "CEA", "AIT")):
            cdMatrix = makeCdMatrix(scale=self.scale, orientation=orientation, flipX=flipX)
            metadata = makeSimpleWcsMetadata(crpix=self.crpix, crval=crval, cdMatrix=cdMatrix,
                                             projection=projection)
            header = makeLimitedFitsHeader(metadata)
            astropyWcs = astropy.wcs.WCS(header)
            skyWcs = makeSkyWcs(crpix=self.crpix, crval=crval, cdMatrix=cdMatrix, projection=projection)
            # Most projections only seem to agree to within 1e-4 in the round trip test
            self.assertSkyWcsAstropyWcsAlmostEqual(skyWcs=skyWcs, astropyWcs=astropyWcs, bbox=bbox)

    def testPixelToSkyArray(self):
        """Test the numpy-array version of pixelToSky
        """
        cdMatrix = makeCdMatrix(scale=self.scale)
        wcs = makeSkyWcs(crpix=self.crpix, crval=self.crvalList[0], cdMatrix=cdMatrix)

        xPoints = np.array([0.0, 1000.0, 0.0, -1111.0])
        yPoints = np.array([0.0, 0.0, 2000.0, -2222.0])

        pixPointList = [lsst.geom.Point2D(x, y) for x, y in zip(xPoints, yPoints)]

        spherePoints = wcs.pixelToSky(pixPointList)

        ra, dec = wcs.pixelToSkyArray(xPoints, yPoints, degrees=False)
        for r, d, spherePoint in zip(ra, dec, spherePoints):
            self.assertAlmostEqual(r, spherePoint.getRa().asRadians())
            self.assertAlmostEqual(d, spherePoint.getDec().asRadians())

        ra, dec = wcs.pixelToSkyArray(xPoints, yPoints, degrees=True)
        for r, d, spherePoint in zip(ra, dec, spherePoints):
            self.assertAlmostEqual(r, spherePoint.getRa().asDegrees())
            self.assertAlmostEqual(d, spherePoint.getDec().asDegrees())

    def testSkyToPixelArray(self):
        """Test the numpy-array version of skyToPixel
        """
        cdMatrix = makeCdMatrix(scale=self.scale)
        wcs = makeSkyWcs(crpix=self.crpix, crval=self.crvalList[0], cdMatrix=cdMatrix)

        raPoints = np.array([3.92646679e-02, 3.59646622e+02,
                             3.96489283e-02, 4.70419353e-01])
        decPoints = np.array([44.9722155, 44.97167735,
                              45.52775599, 44.3540619])

        spherePointList = [lsst.geom.SpherePoint(ra*lsst.geom.degrees,
                                                 dec*lsst.geom.degrees)
                           for ra, dec in zip(raPoints, decPoints)]

        pixPoints = wcs.skyToPixel(spherePointList)

        x, y = wcs.skyToPixelArray(np.deg2rad(raPoints), np.deg2rad(decPoints))
        for x0, y0, pixPoint in zip(x, y, pixPoints):
            self.assertAlmostEqual(x0, pixPoint.getX())
            self.assertAlmostEqual(y0, pixPoint.getY())

        x, y = wcs.skyToPixelArray(raPoints, decPoints, degrees=True)
        for x0, y0, pixPoint in zip(x, y, pixPoints):
            self.assertAlmostEqual(x0, pixPoint.getX())
            self.assertAlmostEqual(y0, pixPoint.getY())

    def testStr(self):
        """Test that we can get something coherent when printing a SkyWcs.
        """
        cdMatrix = makeCdMatrix(scale=self.scale)
        skyWcs = makeSkyWcs(crpix=self.crpix, crval=self.crvalList[0], cdMatrix=cdMatrix)
        self.assertIn(f"Sky Origin: {self.crvalList[0]}", str(skyWcs))
        self.assertIn(f"Pixel Origin: {self.crpix}", str(skyWcs))
        self.assertIn("Pixel Scale: ", str(skyWcs))


class MetadataWcsTestCase(SkyWcsBaseTestCase):
    """Test metadata constructor of SkyWcs
    """

    def setUp(self):
        metadata = PropertyList()
        for name, value in (
            ("RADESYS", "ICRS"),
            ("EQUINOX", 2000.),
            ("CRVAL1", 215.604025685476),
            ("CRVAL2", 53.1595451514076),
            ("CRPIX1", 1109.99981456774),
            ("CRPIX2", 560.018167811613),
            ("CTYPE1", "RA---TAN"),
            ("CTYPE2", "DEC--TAN"),
            ("CUNIT1", "deg"),
            ("CUNIT2", "deg"),
            ("CD1_1", 5.10808596133527E-05),
            ("CD1_2", 1.85579539217196E-07),
            ("CD2_2", -5.10281493481982E-05),
            ("CD2_1", -1.85579539217196E-07),
        ):
            metadata.set(name, value)
        self.metadata = metadata

    def tearDown(self):
        del self.metadata

    def checkWcs(self, skyWcs):
        pixelOrigin = skyWcs.getPixelOrigin()
        skyOrigin = skyWcs.getSkyOrigin()
        for i in range(2):
            # subtract 1 from FITS CRPIX to get LSST convention
            self.assertAlmostEqual(pixelOrigin[i], self.metadata.getScalar(f"CRPIX{i+1}") - 1)
            self.assertAnglesAlmostEqual(skyOrigin[i],
                                         self.metadata.getScalar(f"CRVAL{i+1}")*lsst.geom.degrees)
        cdMatrix = skyWcs.getCdMatrix()
        for i, j in itertools.product(range(2), range(2)):
            self.assertAlmostEqual(cdMatrix[i, j], self.metadata.getScalar(f"CD{i+1}_{j+1}"))

        self.assertTrue(skyWcs.isFits)

        skyWcsCopy = SkyWcs.readString(skyWcs.writeString())
        self.assertTrue(skyWcsCopy.isFits)
        self.checkMakeFlippedWcs(skyWcs)

    @unittest.skipIf(sys.version_info[0] < 3, "astropy.wcs rejects the header on py2")
    def testAgainstAstropyWcs(self):
        skyWcs = makeSkyWcs(self.metadata, strip=False)
        header = makeLimitedFitsHeader(self.metadata)
        astropyWcs = astropy.wcs.WCS(header)
        bbox = lsst.geom.Box2D(lsst.geom.Point2D(-1000, -1000), lsst.geom.Extent2D(3000, 3000))
        self.assertSkyWcsAstropyWcsAlmostEqual(skyWcs=skyWcs, astropyWcs=astropyWcs, bbox=bbox)

    def testLinearizeMethods(self):
        skyWcs = makeSkyWcs(self.metadata)
        # use a sky position near, but not at, the WCS origin
        sky00 = skyWcs.getSkyOrigin().offset(45 * lsst.geom.degrees, 1.2 * lsst.geom.degrees)
        pix00 = skyWcs.skyToPixel(sky00)
        for skyUnit in (lsst.geom.degrees, lsst.geom.radians):
            linPixToSky1 = skyWcs.linearizePixelToSky(sky00, skyUnit)  # should match inverse of linSkyToPix1
            linPixToSky2 = skyWcs.linearizePixelToSky(pix00, skyUnit)  # should match inverse of linSkyToPix1
            linSkyToPix1 = skyWcs.linearizeSkyToPixel(sky00, skyUnit)
            linSkyToPix2 = skyWcs.linearizeSkyToPixel(pix00, skyUnit)  # should match linSkyToPix1

            for pixel in (pix00, pix00 + lsst.geom.Extent2D(1000, -1230)):
                linSky = linPixToSky1(pixel)
                self.assertPairsAlmostEqual(linPixToSky2(pixel), linSky)
                self.assertPairsAlmostEqual(linSkyToPix1(linSky), pixel)
                self.assertPairsAlmostEqual(linSkyToPix2(linSky), pixel)

            sky00Doubles = sky00.getPosition(skyUnit)
            pix00gApprox = linSkyToPix1(sky00Doubles)
            self.assertPairsAlmostEqual(pix00gApprox, pix00)
            self.assertAlmostEqual(pix00.getX(), pix00gApprox.getX())
            self.assertAlmostEqual(pix00.getY(), pix00gApprox.getY())
            pixelScale = skyWcs.getPixelScale(pix00)
            pixelArea = pixelScale.asAngularUnits(skyUnit)**2
            predictedPixelArea = 1 / linSkyToPix1.getLinear().computeDeterminant()
            self.assertAlmostEqual(pixelArea, predictedPixelArea)

    def testBasics(self):
        skyWcs = makeSkyWcs(self.metadata, strip=False)
        self.assertEqual(len(self.metadata.names(False)), 14)
        self.checkWcs(skyWcs)
        makeSkyWcs(self.metadata, strip=True)
        self.assertEqual(len(self.metadata.names(False)), 0)

    def testBasicsStrip(self):
        stripWcsMetadata(self.metadata)
        self.assertEqual(len(self.metadata.names(False)), 0)
        # The metadata should be unchanged if we attempt to strip it again
        metadataCopy = self.metadata.deepCopy()
        stripWcsMetadata(self.metadata)
        for key in self.metadata.keys():
            self.assertEqual(self.metadata[key], metadataCopy[key])

    def testNormalizationFk5(self):
        """Test that readLsstSkyWcs correctly normalizes FK5 1975 to ICRS
        """
        equinox = 1975.0
        metadata = self.metadata

        metadata.set("RADESYS", "FK5")
        metadata.set("EQUINOX", equinox)
        crpix = lsst.geom.Point2D(metadata.getScalar("CRPIX1") - 1, metadata.getScalar("CRPIX2") - 1)
        # record the original CRVAL before reading and stripping metadata
        crvalFk5Deg = (metadata.getScalar("CRVAL1"), metadata.getScalar("CRVAL2"))

        # create the wcs and retrieve crval
        skyWcs = makeSkyWcs(metadata)
        crval = skyWcs.getSkyOrigin()

        # compare to computed crval
        computedCrval = skyWcs.pixelToSky(crpix)
        self.assertSpherePointsAlmostEqual(crval, computedCrval)

        # get predicted crval by converting with astropy
        crvalFk5 = astropy.coordinates.SkyCoord(crvalFk5Deg[0], crvalFk5Deg[1], frame="fk5",
                                                equinox=f"J{equinox}", unit="deg")
        predictedCrvalIcrs = crvalFk5.icrs
        predictedCrval = lsst.geom.SpherePoint(predictedCrvalIcrs.ra.radian, predictedCrvalIcrs.dec.radian,
                                               lsst.geom.radians)
        self.assertSpherePointsAlmostEqual(crval, predictedCrval, maxSep=0.002*lsst.geom.arcseconds)

    def testNormalizationDecRa(self):
        """Test that a Dec, RA WCS is normalized to RA, Dec
        """
        crpix = lsst.geom.Point2D(self.metadata.getScalar("CRPIX1") - 1,
                                  self.metadata.getScalar("CRPIX2") - 1)

        # swap RA, Decaxes in metadata
        crvalIn = lsst.geom.SpherePoint(self.metadata.getScalar("CRVAL1"),
                                        self.metadata.getScalar("CRVAL2"), lsst.geom.degrees)
        self.metadata.set("CRVAL1", crvalIn[1].asDegrees())
        self.metadata.set("CRVAL2", crvalIn[0].asDegrees())
        self.metadata.set("CTYPE1", "DEC--TAN")
        self.metadata.set("CTYPE2", "RA---TAN")

        # create the wcs
        skyWcs = makeSkyWcs(self.metadata)

        # compare pixel origin to input crval
        crval = skyWcs.getSkyOrigin()
        self.assertSpherePointsAlmostEqual(crval, crvalIn)

        # compare to computed crval
        computedCrval = skyWcs.pixelToSky(crpix)
        self.assertSpherePointsAlmostEqual(crval, computedCrval)

    def testReadDESHeader(self):
        """Verify that we can read a DES header"""
        self.metadata.set("RADESYS", "ICRS   ")  # note trailing white space
        self.metadata.set("CTYPE1", "RA---TPV")
        self.metadata.set("CTYPE2", "DEC--TPV")

        skyWcs = makeSkyWcs(self.metadata, strip=False)
        self.checkWcs(skyWcs)

    def testCD_PC(self):
        """Test that we can read a FITS file with both CD and PC keys (like early Suprimecam files)"""
        md = PropertyList()
        for k, v in (
            ("EQUINOX", 2000.0),
            ("RADESYS", "ICRS"),
            ("CRPIX1", 5353.0),
            ("CRPIX2", -35.0),
            ("CD1_1", 0.0),
            ("CD1_2", -5.611E-05),
            ("CD2_1", -5.611E-05),
            ("CD2_2", -0.0),
            ("CRVAL1", 4.5789875),
            ("CRVAL2", 16.30004444),
            ("CUNIT1", "deg"),
            ("CUNIT2", "deg"),
            ("CTYPE1", "RA---TAN"),
            ("CTYPE2", "DEC--TAN"),
            ("CDELT1", -5.611E-05),
            ("CDELT2", 5.611E-05),
        ):
            md.set(k, v)

        wcs = makeSkyWcs(md, strip=False)

        pixPos = lsst.geom.Point2D(1000, 2000)
        pred_skyPos = lsst.geom.SpherePoint(4.459815023498577, 16.544199850984768, lsst.geom.degrees)

        skyPos = wcs.pixelToSky(pixPos)
        self.assertSpherePointsAlmostEqual(skyPos, pred_skyPos)

        for badPC in (False, True):
            for k, v in (
                ("PC001001", 0.0),
                ("PC001002", -1.0 if badPC else 1.0),
                ("PC002001", 1.0 if badPC else -1.0),
                ("PC002002", 0.0),
            ):
                md.set(k, v)

            # Check Greisen and Calabretta A&A 395 1061 (2002), Eq. 3
            if not badPC:
                for i in (1, 2,):
                    for j in (1, 2,):
                        self.assertEqual(md.getScalar(f"CD{i}_{j}"),
                                         md.getScalar(f"CDELT{i}")*md.getScalar(f"PC00{i}00{j}"))

            wcs2 = makeSkyWcs(md, strip=False)
            skyPos2 = wcs2.pixelToSky(pixPos)
            self.assertSpherePointsAlmostEqual(skyPos2, pred_skyPos)

    def testNoEpoch(self):
        """Ensure we're not writing epoch headers (DATE-OBS, MJD-OBS)"""
        self.metadata.set("EQUINOX", 2000.0)  # Triggers AST writing DATE-OBS, MJD-OBS
        skyWcs = makeSkyWcs(self.metadata)
        header = skyWcs.getFitsMetadata()
        self.assertFalse(header.exists("DATE-OBS"))
        self.assertFalse(header.exists("MJD-OBS"))

    def testCdMatrix(self):
        """Ensure we're writing CD matrix elements even if they're zero"""
        self.metadata.remove("CD1_2")
        self.metadata.remove("CD2_1")
        skyWcs = makeSkyWcs(self.metadata, strip=False)
        header = skyWcs.getFitsMetadata()
        for keyword in ("CD1_1", "CD2_2"):
            # There's some mild rounding going on
            self.assertFloatsAlmostEqual(header.get(keyword), self.metadata.get(keyword), atol=1.0e-16)
        for keyword in ("CD1_2", "CD2_1"):
            self.assertTrue(header.exists(keyword))
            self.assertEqual(header.get(keyword), 0.0)


class TestTanSipTestCase(SkyWcsBaseTestCase):

    def setUp(self):
        metadata = PropertyList()
        # the following was fit using CreateWcsWithSip from meas_astrom
        # and is valid over this bbox: (minimum=(0, 0), maximum=(3030, 3030))
        # This same metadata was used to create testdata/oldTanSipwWs.fits
        for name, value in (
            ("RADESYS", "ICRS"),
            ("CTYPE1", "RA---TAN-SIP"),
            ("CTYPE2", "DEC--TAN-SIP"),
            ("CRPIX1", 1531.1824767147),
            ("CRPIX2", 1531.1824767147),
            ("CRVAL1", 43.035511801383),
            ("CRVAL2", 44.305697682784),
            ("CUNIT1", "deg"),
            ("CUNIT2", "deg"),
            ("CD1_1", 0.00027493991598151),
            ("CD1_2", -3.2758487104158e-06),
            ("CD2_1", 3.2301310675830e-06),
            ("CD2_2", 0.00027493937506632),
            ("A_ORDER", 5),
            ("A_0_2", -1.7769487466972e-09),
            ("A_0_3", 5.3745894718340e-13),
            ("A_0_4", -7.2921116596880e-17),
            ("A_0_5", 8.6947236956136e-21),
            ("A_1_1", 5.4246387438098e-08),
            ("A_1_2", -1.5689083084641e-12),
            ("A_1_3", 1.2424130500997e-16),
            ("A_1_4", 3.9982572658006e-20),
            ("A_2_0", 4.9268299826160e-08),
            ("A_2_1", 1.6365657558495e-12),
            ("A_2_2", 1.1976983061953e-16),
            ("A_2_3", -1.7262037266467e-19),
            ("A_3_0", -5.9235031179999e-13),
            ("A_3_1", -3.4444326387310e-16),
            ("A_3_2", 1.4377441160800e-19),
            ("A_4_0", 1.8736407845095e-16),
            ("A_4_1", 2.9213314172884e-20),
            ("A_5_0", -5.3601346091084e-20),
            ("B_ORDER", 5),
            ("B_0_2", 4.9268299822979e-08),
            ("B_0_3", -5.9235032026906e-13),
            ("B_0_4", 1.8736407776035e-16),
            ("B_0_5", -5.3601341373220e-20),
            ("B_1_1", 5.4246387435453e-08),
            ("B_1_2", 1.6365657531115e-12),
            ("B_1_3", -3.4444326228808e-16),
            ("B_1_4", 2.9213312399941e-20),
            ("B_2_0", -1.7769487494962e-09),
            ("B_2_1", -1.5689082999319e-12),
            ("B_2_2", 1.1976983393279e-16),
            ("B_2_3", 1.4377441169892e-19),
            ("B_3_0", 5.3745894237186e-13),
            ("B_3_1", 1.2424130479929e-16),
            ("B_3_2", -1.7262036838229e-19),
            ("B_4_0", -7.2921117326608e-17),
            ("B_4_1", 3.9982566975450e-20),
            ("B_5_0", 8.6947240592408e-21),
            ("AP_ORDER", 6),
            ("AP_0_0", -5.4343024221207e-11),
            ("AP_0_1", 5.5722265946666e-12),
            ("AP_0_2", 1.7769484042400e-09),
            ("AP_0_3", -5.3773609554820e-13),
            ("AP_0_4", 7.3035278852156e-17),
            ("AP_0_5", -8.7151153799062e-21),
            ("AP_0_6", 3.2535945427624e-27),
            ("AP_1_0", -3.8944805432871e-12),
            ("AP_1_1", -5.4246388067582e-08),
            ("AP_1_2", 1.5741716194971e-12),
            ("AP_1_3", -1.2447067748187e-16),
            ("AP_1_4", -3.9960260822306e-20),
            ("AP_1_5", 1.1297941471380e-26),
            ("AP_2_0", -4.9268299293185e-08),
            ("AP_2_1", -1.6256111849359e-12),
            ("AP_2_2", -1.1973373130440e-16),
            ("AP_2_3", 1.7266948205700e-19),
            ("AP_2_4", -3.7059606160753e-26),
            ("AP_3_0", 5.9710911995811e-13),
            ("AP_3_1", 3.4464427650041e-16),
            ("AP_3_2", -1.4381853884204e-19),
            ("AP_3_3", -7.6527426974322e-27),
            ("AP_4_0", -1.8748435698960e-16),
            ("AP_4_1", -2.9267280226373e-20),
            ("AP_4_2", 4.8004317051259e-26),
            ("AP_5_0", 5.3657330221120e-20),
            ("AP_5_1", -1.6904065766661e-27),
            ("AP_6_0", -1.9484495120493e-26),
            ("BP_ORDER", 6),
            ("BP_0_0", -5.4291220607725e-11),
            ("BP_0_1", -3.8944871307931e-12),
            ("BP_0_2", -4.9268299290361e-08),
            ("BP_0_3", 5.9710912831833e-13),
            ("BP_0_4", -1.8748435594265e-16),
            ("BP_0_5", 5.3657325543368e-20),
            ("BP_0_6", -1.9484577299247e-26),
            ("BP_1_0", 5.5722051513577e-12),
            ("BP_1_1", -5.4246388065000e-08),
            ("BP_1_2", -1.6256111821465e-12),
            ("BP_1_3", 3.4464427499767e-16),
            ("BP_1_4", -2.9267278448109e-20),
            ("BP_1_5", -1.6904244067295e-27),
            ("BP_2_0", 1.7769484069376e-09),
            ("BP_2_1", 1.5741716110182e-12),
            ("BP_2_2", -1.1973373446176e-16),
            ("BP_2_3", -1.4381853893526e-19),
            ("BP_2_4", 4.8004294492911e-26),
            ("BP_3_0", -5.3773609074713e-13),
            ("BP_3_1", -1.2447067726801e-16),
            ("BP_3_2", 1.7266947774875e-19),
            ("BP_3_3", -7.6527556667042e-27),
            ("BP_4_0", 7.3035279660505e-17),
            ("BP_4_1", -3.9960255158200e-20),
            ("BP_4_2", -3.7059659675039e-26),
            ("BP_5_0", -8.7151157361284e-21),
            ("BP_5_1", 1.1297944388060e-26),
            ("BP_6_0", 3.2535788867488e-27),
        ):
            metadata.set(name, value)
        self.metadata = metadata
        self.bbox = lsst.geom.Box2D(lsst.geom.Point2D(-1000, -1000), lsst.geom.Extent2D(3000, 3000))

    def testTanSipFromFrameDict(self):
        """Test making a TAN-SIP WCS from a FrameDict
        """
        skyWcs = makeSkyWcs(self.metadata, strip=False)
        self.checkFrameDictConstructor(skyWcs, bbox=self.bbox)

    def testFitsMetadata(self):
        """Test that getFitsMetadata works for TAN-SIP
        """
        skyWcs = makeSkyWcs(self.metadata, strip=False)
        self.assertTrue(skyWcs.isFits)
        fitsMetadata = skyWcs.getFitsMetadata(precise=True)
        skyWcsCopy = makeSkyWcs(fitsMetadata)
        self.assertWcsAlmostEqualOverBBox(skyWcs, skyWcsCopy, self.bbox)
        self.checkPersistence(skyWcs, bbox=self.bbox)

    def testGetIntermediateWorldCoordsToSky(self):
        """Test getIntermediateWorldCoordsToSky and getPixelToIntermediateWorldCoords
        """
        crpix = lsst.geom.Extent2D(self.metadata.getScalar("CRPIX1") - 1,
                                   self.metadata.getScalar("CRPIX2") - 1)
        skyWcs = makeSkyWcs(self.metadata, strip=False)
        for simplify in (False, True):
            pixelToIwc = getPixelToIntermediateWorldCoords(skyWcs, simplify)
            iwcToSky = getIntermediateWorldCoordsToSky(skyWcs, simplify)
            self.assertTrue(isinstance(pixelToIwc, TransformPoint2ToPoint2))
            self.assertTrue(isinstance(iwcToSky, TransformPoint2ToSpherePoint))
            if simplify:
                self.assertTrue(pixelToIwc.getMapping().isSimple)
                self.assertTrue(iwcToSky.getMapping().isSimple)
            # else the mapping may have already been simplified inside the WCS,
            # so don't assert isSimple is false

            # check that the chained transforms produce the same results as the WCS
            # in the forward and inverse direction
            pixPosList = []
            for dx in (0, 1000):
                for dy in (0, 1000):
                    pixPosList.append(lsst.geom.Point2D(dx, dy) + crpix)
            iwcPosList = pixelToIwc.applyForward(pixPosList)
            skyPosList = iwcToSky.applyForward(iwcPosList)
            self.assertSpherePointListsAlmostEqual(skyPosList, skyWcs.pixelToSky(pixPosList))
            self.assertPairListsAlmostEqual(pixelToIwc.applyInverse(iwcToSky.applyInverse(skyPosList)),
                                            skyWcs.skyToPixel(skyPosList))

            self.assertPairListsAlmostEqual(iwcPosList, iwcToSky.applyInverse(skyPosList))
            self.assertPairListsAlmostEqual(pixPosList, pixelToIwc.applyInverse(iwcPosList))

            # compare extracted pixelToIwc to a version of pixelToIwc computed directly from the metadata
            ourPixelToIwc = makeSipPixelToIwc(self.metadata)
            self.assertPairListsAlmostEqual(pixelToIwc.applyForward(pixPosList),
                                            ourPixelToIwc.applyForward(pixPosList))

            # compare extracted iwcToPixel to a version of iwcToPixel computed directly from the metadata
            ourIwcToPixel = makeSipIwcToPixel(self.metadata)
            self.assertPairListsAlmostEqual(pixelToIwc.applyInverse(iwcPosList),
                                            ourIwcToPixel.applyForward(iwcPosList))

    @unittest.skipIf(sys.version_info[0] < 3, "astropy.wcs rejects the header on py2")
    def testAgainstAstropyWcs(self):
        skyWcs = makeSkyWcs(self.metadata, strip=False)
        header = makeLimitedFitsHeader(self.metadata)
        astropyWcs = astropy.wcs.WCS(header)
        self.assertSkyWcsAstropyWcsAlmostEqual(skyWcs=skyWcs, astropyWcs=astropyWcs, bbox=self.bbox)

    def testMakeTanSipWcs(self):
        referenceWcs = makeSkyWcs(self.metadata, strip=False)

        crpix = lsst.geom.Point2D(self.metadata.getScalar("CRPIX1") - 1,
                                  self.metadata.getScalar("CRPIX2") - 1)
        crval = lsst.geom.SpherePoint(self.metadata.getScalar("CRVAL1"),
                                      self.metadata.getScalar("CRVAL2"), lsst.geom.degrees)
        cdMatrix = getCdMatrixFromMetadata(self.metadata)
        sipA = getSipMatrixFromMetadata(self.metadata, "A")
        sipB = getSipMatrixFromMetadata(self.metadata, "B")
        sipAp = getSipMatrixFromMetadata(self.metadata, "AP")
        sipBp = getSipMatrixFromMetadata(self.metadata, "BP")
        skyWcs1 = makeTanSipWcs(crpix=crpix, crval=crval, cdMatrix=cdMatrix, sipA=sipA, sipB=sipB)
        skyWcs2 = makeTanSipWcs(crpix=crpix, crval=crval, cdMatrix=cdMatrix, sipA=sipA, sipB=sipB,
                                sipAp=sipAp, sipBp=sipBp)

        self.assertWcsAlmostEqualOverBBox(referenceWcs, skyWcs1, self.bbox)
        self.assertWcsAlmostEqualOverBBox(referenceWcs, skyWcs2, self.bbox)
        self.checkMakeFlippedWcs(skyWcs1)
        self.checkMakeFlippedWcs(skyWcs2)

    def testReadWriteFits(self):
        wcsFromMetadata = makeSkyWcs(self.metadata)
        with lsst.utils.tests.getTempFilePath(".fits") as filePath:
            wcsFromMetadata.writeFits(filePath)
            wcsFromFits = SkyWcs.readFits(filePath)

        self.assertWcsAlmostEqualOverBBox(wcsFromFits, wcsFromMetadata, self.bbox, maxDiffPix=0,
                                          maxDiffSky=0*lsst.geom.radians)

    def testReadOldTanSipFits(self):
        """Test reading a FITS file containing data for an lsst::afw::image::TanWcs

        That file was made using the same metadata as this test
        """
        dataDir = os.path.join(os.path.split(__file__)[0], "data")
        filePath = os.path.join(dataDir, "oldTanSipWcs.fits")
        wcsFromFits = SkyWcs.readFits(filePath)

        wcsFromMetadata = makeSkyWcs(self.metadata)

        bbox = lsst.geom.Box2D(lsst.geom.Point2D(-1000, -1000), lsst.geom.Extent2D(3000, 3000))
        self.assertWcsAlmostEqualOverBBox(wcsFromFits, wcsFromMetadata, bbox)

    def testReadOldTanFits(self):
        """Test reading a FITS file containing data for an lsst::afw::image::TanWcs

        That file was made using the same metadata follows
        (like self.metadata without the distortion)
        """
        tanMetadata = PropertyList()
        # the following was fit using CreateWcsWithSip from meas_astrom
        # and is valid over this bbox: (minimum=(0, 0), maximum=(3030, 3030))
        # This same metadata was used to create testdata/oldTanSipwWs.fits
        for name, value in (
            ("RADESYS", "ICRS"),
            ("CTYPE1", "RA---TAN"),
            ("CTYPE2", "DEC--TAN"),
            ("CRPIX1", 1531.1824767147),
            ("CRPIX2", 1531.1824767147),
            ("CRVAL1", 43.035511801383),
            ("CRVAL2", 44.305697682784),
            ("CUNIT1", "deg"),
            ("CUNIT2", "deg"),
            ("CD1_1", 0.00027493991598151),
            ("CD1_2", -3.2758487104158e-06),
            ("CD2_1", 3.2301310675830e-06),
            ("CD2_2", 0.00027493937506632),
        ):
            tanMetadata.set(name, value)

        dataDir = os.path.join(os.path.split(__file__)[0], "data")
        filePath = os.path.join(dataDir, "oldTanWcs.fits")
        wcsFromFits = SkyWcs.readFits(filePath)

        wcsFromMetadata = makeSkyWcs(tanMetadata)

        bbox = lsst.geom.Box2D(lsst.geom.Point2D(-1000, -1000), lsst.geom.Extent2D(3000, 3000))
        self.assertWcsAlmostEqualOverBBox(wcsFromFits, wcsFromMetadata, bbox)


class WcsPairTransformTestCase(SkyWcsBaseTestCase):
    """Test functionality of makeWcsPairTransform.
    """
    def setUp(self):
        SkyWcsBaseTestCase.setUp(self)
        crpix = lsst.geom.Point2D(100, 100)
        crvalList = [
            lsst.geom.SpherePoint(0, 45, lsst.geom.degrees),
            lsst.geom.SpherePoint(0.00001, 45, lsst.geom.degrees),
            lsst.geom.SpherePoint(359.99999, 45, lsst.geom.degrees),
            lsst.geom.SpherePoint(30, 89.99999, lsst.geom.degrees),
        ]
        orientationList = [
            0 * lsst.geom.degrees,
            0.00001 * lsst.geom.degrees,
            -0.00001 * lsst.geom.degrees,
            -45 * lsst.geom.degrees,
            90 * lsst.geom.degrees,
        ]
        scale = 1.0 * lsst.geom.arcseconds

        self.wcsList = []
        for crval in crvalList:
            for orientation in orientationList:
                cd = makeCdMatrix(scale=scale, orientation=orientation)
                self.wcsList.append(makeSkyWcs(
                    crpix=crpix,
                    crval=crval,
                    cdMatrix=cd))
        self.pixelPoints = [lsst.geom.Point2D(x, y) for x, y in
                            itertools.product((0.0, -2.0, 42.5, 1042.3),
                                              (27.6, -0.1, 0.0, 196.0))]

    def testGenericWcs(self):
        """Test that input and output points represent the same sky position.

        Would prefer a black-box test, but don't have the numbers for it.
        """
        inPoints = self.pixelPoints
        for wcs1 in self.wcsList:
            for wcs2 in self.wcsList:
                transform = makeWcsPairTransform(wcs1, wcs2)
                outPoints = transform.applyForward(inPoints)
                inPointsRoundTrip = transform.applyInverse(outPoints)
                self.assertPairListsAlmostEqual(inPoints, inPointsRoundTrip)
                self.assertSpherePointListsAlmostEqual(wcs1.pixelToSky(inPoints),
                                                       wcs2.pixelToSky(outPoints))

    def testSameWcs(self):
        """Confirm that pairing two identical Wcs gives an identity transform.
        """
        for wcs in self.wcsList:
            transform = makeWcsPairTransform(wcs, wcs)
            # check that the transform has been simplified
            self.assertTrue(transform.getMapping().isSimple)
            # check the transform
            outPoints1 = transform.applyForward(self.pixelPoints)
            outPoints2 = transform.applyInverse(outPoints1)
            self.assertPairListsAlmostEqual(self.pixelPoints, outPoints1)
            self.assertPairListsAlmostEqual(outPoints1, outPoints2)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
