from __future__ import absolute_import, division, print_function
import itertools
import os
import sys
import unittest

import astropy.io.fits
import astropy.coordinates
import astropy.wcs
from numpy.testing import assert_allclose

import lsst.utils.tests
from lsst.daf.base import PropertyList
from lsst.afw.coord import IcrsCoord
from lsst.afw.fits import readMetadata
from lsst.afw.geom import Extent2D, Point2D, Extent2I, Point2I, \
    Box2I, Box2D, degrees, arcseconds, radians, wcsAlmostEqualOverBBox, \
    TransformPoint2ToPoint2, TransformPoint2ToIcrsCoord, makeRadialTransform, \
    SkyWcs, makeSkyWcs, makeCdMatrix, makeWcsPairTransform, \
    makeFlippedWcs, makeModifiedWcs, makeTanSipWcs, \
    getIntermediateWorldCoordsToSky, getPixelToIntermediateWorldCoords
from lsst.afw.geom.wcsUtils import getCdMatrixFromMetadata, getSipMatrixFromMetadata, makeSimpleWcsMetadata
from lsst.afw.geom.testUtils import makeFitsHeaderFromMetadata, makeSipIwcToPixel, makeSipPixelToIwc


class SkyWcsBaseTestCase(lsst.utils.tests.TestCase):
    def checkPersistence(self, skyWcs):
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
            Point2D(0, 0),
            Point2D(1000, 0),
            Point2D(0, 1000),
            Point2D(-50, -50),
        ]
        skyPoints = skyWcs.pixelToSky(pixelPoints)
        pixelPoints2 = skyWcs.skyToPixel(skyPoints)
        assert_allclose(pixelPoints, pixelPoints2, atol=1e-7)

    def checkMakeFlippedWcs(self, skyWcs, skyAtol=1e-7*arcseconds):
        """Check makeFlippedWcs on the provided WCS
        """
        # make an arbitrary bbox, but one that includes zero in one axis
        # and does not include zero in the other axis
        # the center of the bbox is used as the center of flipping
        # and the corners of the bbox are the input positions that are tested
        bbox = Box2D(Point2D(-100, 1000), Extent2D(2000, 1501))
        # dict of (isRight, isTop): position
        minPos = bbox.getMin()
        maxPos = bbox.getMax()
        center = bbox.getCenter()
        cornerDict = {
            (False, False): minPos,
            (False, True): Point2D(minPos[0], maxPos[1]),
            (True, False): Point2D(maxPos[0], minPos[1]),
            (True, True): maxPos,
        }
        for flipLR, flipTB in itertools.product((False, True), (False, True)):
            flippedWcs = makeFlippedWcs(wcs=skyWcs, flipLR=flipLR, flipTB=flipTB, center=center)
            # the center is unchanged
            self.assertCoordsAlmostEqual(skyWcs.pixelToSky(center),
                                         flippedWcs.pixelToSky(center), maxDiff=skyAtol)

            for isR, isT in itertools.product((False, True), (False, True)):
                origPos = cornerDict[(isR, isT)]
                flippedPos = cornerDict[(isR ^ flipLR, isT ^ flipTB)]
                self.assertCoordsAlmostEqual(skyWcs.pixelToSky(origPos),
                                             flippedWcs.pixelToSky(flippedPos), maxDiff=skyAtol)

    def assertSkyWcsAstropyWcsAlmostEqual(self, skyWcs, astropyWcs, bbox,
                                          pixAtol=1e-4, skyAtol=1e-4*arcseconds, checkRoundTrip=True):
        """Assert that a SkyWcs and the corresponding astropy.wcs.WCS agree over a specified bounding box
        """
        bbox = Box2D(bbox)
        center = bbox.getCenter()
        xArr = bbox.getMinX(), center[0], bbox.getMaxX()
        yArr = bbox.getMinY(), center[1], bbox.getMaxY()
        pixPosList = [Point2D(x, y) for x, y in itertools.product(xArr, yArr)]

        # pixelToSky
        skyPosList = skyWcs.pixelToSky(pixPosList)
        astropySkyPosList = self.astropyPixelsToSky(astropyWcs=astropyWcs, pixPosList=pixPosList)
        self.assertCoordListsAlmostEqual(skyPosList, astropySkyPosList, maxDiff=skyAtol)

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
        @param[in] pixPosList 0-based pixel positions as lsst.afw.geom.Point2D or similar pairs
        @returns sky coordinates as a list of lsst.afw.coord.IcrsCoord

        Converts the output to ICRS
        """
        xarr = [p[0] for p in pixPosList]
        yarr = [p[1] for p in pixPosList]
        skyCoordList = astropy.wcs.utils.pixel_to_skycoord(xp = xarr,
                                                           yp = yarr,
                                                           wcs = astropyWcs,
                                                           origin = 0,
                                                           mode = "all")
        icrsList = [sc.transform_to("icrs") for sc in skyCoordList]
        return [IcrsCoord(sc.ra.deg * degrees, sc.dec.deg * degrees) for sc in icrsList]

    def astropySkyToPixels(self, astropyWcs, skyPosList):
        """Use an astropy wcs to convert pixels to sky

        @param[in] astropyWcs  a celestial astropy.wcs.WCS with 2 axes in RA, Dec order
        @param[in] skyPosList ICRS sky coordinates as a list of lsst.afw.coord.IcrsCoord
        @returns a list of lsst.afw.geom.Point2D, 0-based pixel positions

        Converts the input from ICRS to the coordinate system of the wcs
        """
        skyCoordList = [astropy.coordinates.SkyCoord(c[0].asDegrees(),
                                                     c[1].asDegrees(),
                                                     frame="icrs",
                                                     unit = "deg") for c in skyPosList]
        xyArr = [astropy.wcs.utils.skycoord_to_pixel(coords = sc,
                                                     wcs = astropyWcs,
                                                     origin = 0,
                                                     mode = "all") for sc in skyCoordList]
        # float is needed to avoid truncation to int
        return [Point2D(float(x), float(y)) for x, y in xyArr]


class SimpleSkyWcsTestCase(SkyWcsBaseTestCase):
    """Test the simple FITS version of makeSkyWcs
    """

    def setUp(self):
        self.crpix = Point2D(100, 100)
        self.crvalList = [
            IcrsCoord(0 * degrees, 45 * degrees),
            IcrsCoord(0.00001 * degrees, 45 * degrees),
            IcrsCoord(359.99999 * degrees, 45 * degrees),
            IcrsCoord(30 * degrees, 89.99999 * degrees),
            IcrsCoord(30 * degrees, -89.99999 * degrees),
        ]
        self.orientationList = [
            0 * degrees,
            0.00001 * degrees,
            -0.00001 * degrees,
            -45 * degrees,
            90 * degrees,
        ]
        self.scale = 1.0 * arcseconds
        self.tinyPixels = 1.0e-10
        self.tinyAngle = 1.0e-10 * radians

    def checkTanWcs(self, crval, orientation, flipX):
        """Construct a pure TAN SkyWcs and check that it operates as specified

        Parameters
        ----------
        crval : `lsst.afw.coord.IcrsCoord`
            Desired reference sky position.
            Must not be at either pole.
        orientation : `lsst.afw.geom.Angle`
            Position angle of focal plane +Y, measured from N through E.
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
        self.checkPersistence(wcs)
        self.checkMakeFlippedWcs(wcs)

        self.assertTrue(wcs.isFits)
        self.assertEqual(wcs.isFlipped, bool(flipX))

        xoffAng = 0*degrees if flipX else 180*degrees

        def safeOffset(crval, direction, amount):
            try:
                offsetCoord = crval.toIcrs()
                offsetCoord.offset(direction, amount)
                return offsetCoord
            except Exception:
                return IcrsCoord(crval[0] + direction, crval - amount)

        pixelList = [
            Point2D(self.crpix[0], self.crpix[1]),
            Point2D(self.crpix[0] + 1, self.crpix[1]),
            Point2D(self.crpix[0], self.crpix[1] + 1),
        ]
        skyList = wcs.pixelToSky(pixelList)

        # check pixels to sky
        predSkyList = [
            crval,
            safeOffset(crval, xoffAng - orientation, self.scale),
            safeOffset(crval, 90*degrees - orientation, self.scale),
        ]
        self.assertCoordListsAlmostEqual(predSkyList, skyList)
        self.assertCoordListsAlmostEqual(predSkyList, wcs.pixelToSky(pixelList))
        for pixel, predSky in zip(pixelList, predSkyList):
            self.assertCoordsAlmostEqual(predSky, wcs.pixelToSky(pixel))
            self.assertCoordsAlmostEqual(predSky, wcs.pixelToSky(pixel[0], pixel[1]))

        # check sky to pixels
        self.assertPairListsAlmostEqual(pixelList, wcs.skyToPixel(skyList))
        self.assertPairListsAlmostEqual(pixelList, wcs.skyToPixel(skyList))
        for pixel, sky in zip(pixelList, skyList):
            self.assertPairsAlmostEqual(pixel, wcs.skyToPixel(sky))
            # self.assertPairsAlmostEqual(pixel, wcs.skyToPixel(sky[0], sky[1]))

        # check CRVAL round trip
        self.assertCoordsAlmostEqual(wcs.getSkyOrigin(), crval,
                                     maxDiff=self.tinyAngle)

        crpix = wcs.getPixelOrigin()
        self.assertPairsAlmostEqual(crpix, self.crpix, maxDiff=self.tinyPixels)

        self.assertFloatsAlmostEqual(wcs.getCdMatrix(), cdMatrix)

        pixelScale = wcs.getPixelScale()
        self.assertAnglesAlmostEqual(self.scale, pixelScale, maxDiff=self.tinyAngle)

        pixelScale = wcs.getPixelScale(self.crpix)
        self.assertAnglesAlmostEqual(self.scale, pixelScale, maxDiff=self.tinyAngle)

        # Compute a WCS with the pixel origin shifted by an arbitrary amount
        # The resulting sky origin should not change
        offset = Extent2D(500, -322)  # arbitrary
        shiftedWcs = wcs.copyAtShiftedPixelOrigin(offset)
        self.assertTrue(shiftedWcs.isFits)
        predShiftedPixelOrigin = self.crpix + offset
        self.assertPairsAlmostEqual(shiftedWcs.getPixelOrigin(), predShiftedPixelOrigin,
                                    maxDiff=self.tinyPixels)
        self.assertCoordsAlmostEqual(shiftedWcs.getSkyOrigin(), crval, maxDiff=self.tinyAngle)

        shiftedPixelList = [p + offset for p in pixelList]
        shiftedSkyList = shiftedWcs.pixelToSky(shiftedPixelList)
        self.assertCoordListsAlmostEqual(skyList, shiftedSkyList, maxDiff=self.tinyAngle)

        # Check that the shifted WCS can be round tripped as FITS metadata
        shiftedMetadata = shiftedWcs.getFitsMetadata(True)
        shiftedWcsCopy = makeSkyWcs(shiftedMetadata)
        shiftedBBox = Box2D(predShiftedPixelOrigin, predShiftedPixelOrigin + Extent2I(2000, 2000))
        self.assertWcsAlmostEqualOverBBox(shiftedWcs, shiftedWcsCopy, shiftedBBox)

        wcsCopy = SkyWcs.readString(wcs.writeString())
        self.assertTrue(wcsCopy.isFits)

        return wcs

    def testTanWcs(self):
        """Check a variety of TanWcs, with crval not at a pole.
        """
        for crval, orientation, flipX in itertools.product(self.crvalList,
                                                           self.orientationList,
                                                           (False, True)):
            self.checkTanWcs(crval = crval,
                             orientation = orientation,
                             flipX = flipX,
                             )

    def testMakeModifiedWcs(self):
        cdMatrix = makeCdMatrix(scale=self.scale)
        wcs = makeSkyWcs(crpix=self.crpix, crval=self.crvalList[0], cdMatrix=cdMatrix)
        pixelTransform = makeRadialTransform([0.0, 1.0, 0.0, 0.0011])  # arbitrary but reasonable
        modifiedWcs = makeModifiedWcs(pixelTransform=pixelTransform, wcs=wcs, modifyActualPixels=False)
        equivalentTransform = pixelTransform.then(wcs.getTransform())
        for pixPoint in (  # arbitrary but reasonable
            Point2D(0.0, 0.0),
            Point2D(1000.0, 0.0),
            Point2D(0.0, 2000.0),
            Point2D(-1111.0, -2222.0),
        ):
            outSky = modifiedWcs.pixelToSky(pixPoint)
            desiredOutSky = equivalentTransform.applyForward(pixPoint)
            self.assertCoordsAlmostEqual(outSky, desiredOutSky)

    @unittest.skipIf(sys.version_info[0] < 3, "astropy.wcs rejects the header on py2")
    def testAgainstAstropyWcs(self):
        bbox = Box2D(Point2D(-1000, -1000), Extent2D(2000, 2000))
        for crval, orientation, flipX, projection in itertools.product(self.crvalList,
                                                                       self.orientationList,
                                                                       (False, True),
                                                                       ("TAN", "STG", "CEA", "AIT")):
            cdMatrix = makeCdMatrix(scale=self.scale, orientation=orientation, flipX=flipX)
            metadata = makeSimpleWcsMetadata(crpix=self.crpix, crval=crval, cdMatrix=cdMatrix,
                                             projection=projection)
            header = makeFitsHeaderFromMetadata(metadata)
            astropyWcs = astropy.wcs.WCS(header)
            skyWcs = makeSkyWcs(crpix=self.crpix, crval=crval, cdMatrix=cdMatrix, projection=projection)
            # Most projections only seem to agree to within 1e-4 in the round trip test
            self.assertSkyWcsAstropyWcsAlmostEqual(skyWcs=skyWcs, astropyWcs=astropyWcs, bbox=bbox)


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
            self.assertAlmostEqual(pixelOrigin[i], self.metadata.get("CRPIX%s" % (i+1,)) - 1)
            self.assertAnglesAlmostEqual(skyOrigin[i], self.metadata.get("CRVAL%s" % (i+1,)) * degrees)
        cdMatrix = skyWcs.getCdMatrix()
        for i, j in itertools.product(range(2), range(2)):
            self.assertAlmostEqual(cdMatrix[i, j], self.metadata.get("CD%s_%s" % (i+1, j+1)))

        self.assertTrue(skyWcs.isFits)

        skyWcsCopy = SkyWcs.readString(skyWcs.writeString())
        self.assertTrue(skyWcsCopy.isFits)
        self.checkMakeFlippedWcs(skyWcs)

    @unittest.skipIf(sys.version_info[0] < 3, "astropy.wcs rejects the header on py2")
    def testAgainstAstropyWcs(self):
        skyWcs = makeSkyWcs(self.metadata, strip = False)
        header = makeFitsHeaderFromMetadata(self.metadata)
        astropyWcs = astropy.wcs.WCS(header)
        bbox = Box2D(Point2D(-1000, -1000), Extent2D(3000, 3000))
        self.assertSkyWcsAstropyWcsAlmostEqual(skyWcs=skyWcs, astropyWcs=astropyWcs, bbox=bbox)

    def testLinearizeMethods(self):
        skyWcs = makeSkyWcs(self.metadata)
        # use a sky position near, but not at, the WCS origin
        sky00 = skyWcs.getSkyOrigin()
        sky00.offset(45 * degrees, 1.2 * degrees)
        pix00 = skyWcs.skyToPixel(sky00)
        for skyUnit in (degrees, radians):
            linPixToSky1 = skyWcs.linearizePixelToSky(sky00, skyUnit)  # should match inverse of linSkyToPix1
            linPixToSky2 = skyWcs.linearizePixelToSky(pix00, skyUnit)  # should match inverse of linSkyToPix1
            linSkyToPix1 = skyWcs.linearizeSkyToPixel(sky00, skyUnit)
            linSkyToPix2 = skyWcs.linearizeSkyToPixel(pix00, skyUnit)  # should match linSkyToPix1

            for pixel in (pix00, pix00 + Extent2D(1000, -1230)):
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
        skyWcs = makeSkyWcs(self.metadata, strip = False)
        self.assertEqual(len(self.metadata.names(False)), 14)
        self.checkWcs(skyWcs)
        makeSkyWcs(self.metadata, strip = True)
        self.assertEqual(len(self.metadata.names(False)), 0)

    def testNormalizationFk5(self):
        """Test that readLsstSkyWcs correctly normalizes FK5 1975 to ICRS
        """
        equinox = 1975.0
        metadata = self.metadata

        metadata.set("RADESYS", "FK5")
        metadata.set("EQUINOX", equinox)
        crpix = Point2D(metadata.get("CRPIX1") - 1, metadata.get("CRPIX2") - 1)
        # record the original CRVAL before reading and stripping metadata
        crvalFk5Deg = (metadata.get("CRVAL1"), metadata.get("CRVAL2"))

        # create the wcs and retrieve crval
        skyWcs = makeSkyWcs(metadata)
        crval = skyWcs.getSkyOrigin()

        # compare to computed crval
        computedCrval = skyWcs.pixelToSky(crpix)
        self.assertCoordsAlmostEqual(crval, computedCrval)

        # get predicted crval by converting with astropy
        crvalFk5 = astropy.coordinates.SkyCoord(crvalFk5Deg[0], crvalFk5Deg[1], frame="fk5",
                                                equinox="J%f" % (equinox,), unit="deg")
        predictedCrvalIcrs = crvalFk5.icrs
        predictedCrval = IcrsCoord(predictedCrvalIcrs.ra.radian*radians,
                                   predictedCrvalIcrs.dec.radian*radians)
        self.assertCoordsAlmostEqual(crval, predictedCrval, maxDiff=0.002*arcseconds)

    def testNormalizationDecRa(self):
        """Test that a Dec, RA WCS is normalized to RA, Dec
        """
        crpix = Point2D(self.metadata.get("CRPIX1") - 1, self.metadata.get("CRPIX2") - 1)

        # swap RA, Decaxes in metadata
        crvalIn = IcrsCoord(self.metadata.get("CRVAL1")*degrees,
                            self.metadata.get("CRVAL2")*degrees)
        self.metadata.set("CRVAL1", crvalIn[1].asDegrees())
        self.metadata.set("CRVAL2", crvalIn[0].asDegrees())
        self.metadata.set("CTYPE1", "DEC--TAN")
        self.metadata.set("CTYPE2", "RA---TAN")

        # create the wcs
        skyWcs = makeSkyWcs(self.metadata)

        # compare pixel origin to input crval
        crval = skyWcs.getSkyOrigin()
        self.assertCoordsAlmostEqual(crval, crvalIn)

        # compare to computed crval
        computedCrval = skyWcs.pixelToSky(crpix)
        self.assertCoordsAlmostEqual(crval, computedCrval)

    def testReadDESHeader(self):
        """Verify that we can read a DES header"""
        self.metadata.set("RADESYS", "ICRS   ")  # note trailing white space
        self.metadata.set("CTYPE1", "RA---TPV")
        self.metadata.set("CTYPE2", "DEC--TPV")

        skyWcs = makeSkyWcs(self.metadata, strip = False)
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

        wcs = makeSkyWcs(md, strip = False)

        pixPos = Point2D(1000, 2000)
        pred_skyPos = IcrsCoord(4.459815023498577 * degrees, 16.544199850984768 * degrees)

        skyPos = wcs.pixelToSky(pixPos)
        self.assertCoordsAlmostEqual(skyPos, pred_skyPos)

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
                        self.assertEqual(md.get("CD%d_%d" % (i, j)),
                                         md.get("CDELT%d" % i)*md.get("PC00%d00%d" % (i, j)))

            wcs2 = makeSkyWcs(md, strip = False)
            skyPos2 = wcs2.pixelToSky(pixPos)
            self.assertCoordsAlmostEqual(skyPos2, pred_skyPos)


class TestTanSipTestCase(SkyWcsBaseTestCase):

    def setUp(self):
        metadata = PropertyList()
        # the following was fit using CreateWcsWithSip from meas_astrom
        # and is valid over this bbox: (minimum=(0, 0), maximum=(3030, 3030))
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

    def testGetIntermediateWorldCoordsToSky(self):
        """Test getIntermediateWorldCoordsToSky and getPixelToIntermediateWorldCoords
        """
        crpix = Extent2D(self.metadata.get("CRPIX1") - 1, self.metadata.get("CRPIX2") - 1)
        skyWcs = makeSkyWcs(self.metadata, strip=False)
        for simplify in (False, True):
            pixelToIwc = getPixelToIntermediateWorldCoords(skyWcs, simplify)
            iwcToSky = getIntermediateWorldCoordsToSky(skyWcs, simplify)
            self.assertTrue(isinstance(pixelToIwc, TransformPoint2ToPoint2))
            self.assertTrue(isinstance(iwcToSky, TransformPoint2ToIcrsCoord))
            if simplify:
                self.assertTrue(pixelToIwc.getFrameSet().getMapping().isSimple)
                self.assertTrue(iwcToSky.getFrameSet().getMapping().isSimple)
            # else the mapping may have already been simplified inside the WCS,
            # so don't assert isSimple is false

            # check that the chained transforms produce the same results as the WCS
            # in the forward and inverse direction
            pixPosList = []
            for dx in (0, 1000):
                for dy in (0, 1000):
                    pixPosList.append(Point2D(dx, dy) + crpix)
            iwcPosList = pixelToIwc.applyForward(pixPosList)
            skyPosList = iwcToSky.applyForward(iwcPosList)
            self.assertCoordListsAlmostEqual(skyPosList, skyWcs.pixelToSky(pixPosList))
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
        skyWcs = makeSkyWcs(self.metadata, strip = False)
        header = makeFitsHeaderFromMetadata(self.metadata)
        astropyWcs = astropy.wcs.WCS(header)
        bbox = Box2D(Point2D(-1000, -1000), Extent2D(3000, 3000))
        self.assertSkyWcsAstropyWcsAlmostEqual(skyWcs=skyWcs, astropyWcs=astropyWcs, bbox=bbox)

    def testMakeTanSipWcs(self):
        referenceWcs = makeSkyWcs(self.metadata, strip=False)

        crpix = Point2D(self.metadata.get("CRPIX1") - 1, self.metadata.get("CRPIX2") - 1)
        crval = IcrsCoord(self.metadata.get("CRVAL1") * degrees, self.metadata.get("CRVAL2") * degrees)
        cdMatrix = getCdMatrixFromMetadata(self.metadata)
        sipA = getSipMatrixFromMetadata(self.metadata, "A")
        sipB = getSipMatrixFromMetadata(self.metadata, "B")
        sipAp = getSipMatrixFromMetadata(self.metadata, "AP")
        sipBp = getSipMatrixFromMetadata(self.metadata, "BP")
        skyWcs1 = makeTanSipWcs(crpix=crpix, crval=crval, cdMatrix=cdMatrix, sipA=sipA, sipB=sipB)
        skyWcs2 = makeTanSipWcs(crpix=crpix, crval=crval, cdMatrix=cdMatrix, sipA=sipA, sipB=sipB,
                                sipAp=sipAp, sipBp=sipBp)

        bbox = Box2D(Point2D(-1000, -1000), Extent2D(2000, 2000))
        self.assertWcsAlmostEqualOverBBox(referenceWcs, skyWcs1, bbox)
        self.assertWcsAlmostEqualOverBBox(referenceWcs, skyWcs2, bbox)
        self.checkMakeFlippedWcs(skyWcs1)
        self.checkMakeFlippedWcs(skyWcs2)

    def testReadFits(self):
        dataDir = os.path.join(os.path.split(__file__)[0], "data")
        filePath = os.path.join(dataDir, "wcs-0007358-102.fits")
        hdu = 1

        wcsMetadata = readMetadata(filePath, hdu)
        skyWcs = makeSkyWcs(wcsMetadata)

        with astropy.io.fits.open(filePath) as fitsObj:
            header = fitsObj[1]
            astropyWcs = astropy.wcs.WCS(header)

        # do not check inverse because we don't know the valid region
        bbox = Box2D(Point2D(-1000, -1000), Extent2D(3000, 3000))
        self.assertSkyWcsAstropyWcsAlmostEqual(skyWcs=skyWcs, astropyWcs=astropyWcs, bbox=bbox,
                                               checkRoundTrip=False)


class WcsPairTransformTestCase(SkyWcsBaseTestCase):
    """Test functionality of makeWcsPairTransform.
    """
    def setUp(self):
        SkyWcsBaseTestCase.setUp(self)
        crpix = Point2D(100, 100)
        crvalList = [
            IcrsCoord(0 * degrees, 45 * degrees),
            IcrsCoord(0.00001 * degrees, 45 * degrees),
            IcrsCoord(359.99999 * degrees, 45 * degrees),
            IcrsCoord(30 * degrees, 89.99999 * degrees),
        ]
        orientationList = [
            0 * degrees,
            0.00001 * degrees,
            -0.00001 * degrees,
            -45 * degrees,
            90 * degrees,
        ]
        scale = 1.0 * arcseconds

        self.wcsList = []
        for crval in crvalList:
            for orientation in orientationList:
                cd = makeCdMatrix(scale=scale, orientation=orientation)
                self.wcsList.append(makeSkyWcs(
                    crpix=crpix,
                    crval=crval,
                    cdMatrix=cd))

    def points(self):
        for x in (0.0, -2.0, 42.5, 1042.3):
            for y in (27.6, -0.1, 0.0, 196.0):
                yield Point2D(x, y)

    def testGenericWcs(self):
        """Test that input and output points represent the same sky position.

        Would prefer a black-box test, but don't have the numbers for it.
        """
        for wcs1 in self.wcsList:
            for wcs2 in self.wcsList:
                transform = makeWcsPairTransform(wcs1, wcs2)
                for point1 in self.points():
                    point2 = transform.applyForward(point1)
                    self.assertPairsAlmostEqual(
                        transform.applyInverse(point2),
                        point1)
                    self.assertCoordsAlmostEqual(
                        wcs1.pixelToSky(point1),
                        wcs2.pixelToSky(point2))

    def testSameWcs(self):
        """Confirm that pairing two identical Wcs gives an identity transform.
        """
        for wcs in self.wcsList:
            transform = makeWcsPairTransform(wcs, wcs)
            for point in self.points():
                outPoint1 = transform.applyForward(point)
                outPoint2 = transform.applyInverse(outPoint1)
                self.assertPairsAlmostEqual(point, outPoint1)
                self.assertPairsAlmostEqual(outPoint1, outPoint2)


class GetApproximateFitsWcsTestCase(SkyWcsBaseTestCase):
    def setUp(self):
        self.dataPath = os.path.join(os.path.dirname(__file__), "data")

    def testTanSip(self):
        filePath = os.path.join(self.dataPath, "HSC-0908120-056-small.fits")
        # TODO: DM-10765 replace with the following when Exposure contains a SkyWcs:
        # exposure = ExposureF(filePath)
        # wcs = exposure.getWcs()
        metadata = readMetadata(filePath)
        wcs = SkyWcs(metadata)

        localTanWcs = wcs.getTanWcs(wcs.getPixelOrigin())
        bbox = Box2I(Point2I(0, 0), Extent2I(1000, 1000))  # arbitrary
        self.assertFalse(wcsAlmostEqualOverBBox(wcs, localTanWcs, bbox))

        approxMetadata = wcs.getFitsMetadata(False)
        reconstitutedWcs = makeSkyWcs(approxMetadata)
        self.assertWcsAlmostEqualOverBBox(localTanWcs, reconstitutedWcs, bbox)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
