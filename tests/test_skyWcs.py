from __future__ import absolute_import, division, print_function
import itertools
import os
import sys
import unittest

from astropy.io import fits

import lsst.afw.coord   # needed for assertCoordsAlmostEqual
import lsst.utils.tests
from lsst.daf.base import PropertyList
from lsst.afw.coord import IcrsCoord
from lsst.afw.geom import SkyWcs, Extent2D, Point2D, Extent2I, Point2I, \
    Box2I, degrees, arcseconds, radians, makeCdMatrix, makeWcsPairTransform, \
    getApproximateFitsWcsMetadata, wcsAlmostEqualOverBBox
from lsst.afw.geom.testUtils import TransformTestBaseClass


class TanSkyWcsTestCase(TransformTestBaseClass):
    """Test pure TAN constructor of SkyWcs
    """

    def setUp(self):
        TransformTestBaseClass.setUp(self)
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
        self.tinyPixels = 10 * sys.float_info.epsilon
        self.tinyAngle = 10 * sys.float_info.epsilon * radians

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
        wcs = SkyWcs(
            crpix = self.crpix,
            crval = crval,
            cdMatrix = cdMatrix,
        )
        self.checkPersistence(wcs)

        xoffAng = 180*degrees if flipX else 0*degrees

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
        skyList = wcs.applyForward(pixelList)

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
            anglePair = wcs.pixelToSky(*pixel)
            self.assertCoordsAlmostEqual(predSky, IcrsCoord(*anglePair))

        # check sky to pixels
        self.assertPairListsAlmostEqual(pixelList, wcs.applyInverse(skyList))
        self.assertPairListsAlmostEqual(pixelList, wcs.applyInverse(skyList))
        for pixel, sky in zip(pixelList, skyList):
            self.assertPairsAlmostEqual(pixel, wcs.skyToPixel(sky))
            xyPair = wcs.skyToPixel(*sky)
            self.assertPairsAlmostEqual(pixel, Point2D(*xyPair))

        self.assertCoordsAlmostEqual(wcs.getSkyOrigin(), crval,
                                     maxDiff=self.tinyAngle)

        crpix = wcs.getPixelOrigin()
        self.assertPairsAlmostEqual(crpix, self.crpix, maxDiff=self.tinyPixels)

        self.assertFloatsAlmostEqual(wcs.getCdMatrix(), cdMatrix)

        pixelScale = wcs.getPixelScale(self.crpix)
        self.assertAnglesAlmostEqual(self.scale, pixelScale, maxDiff=self.tinyAngle)

        # Compute a WCS with the pixel origin shifted by an arbitrary amount
        # The resulting sky origin should not change
        offset = Extent2D(500, -322)  # arbitrary
        shiftedWcs = wcs.copyAtShiftedPixelOrigin(offset)
        predShiftedPixelOrigin = self.crpix + offset
        self.assertPairsAlmostEqual(shiftedWcs.getPixelOrigin(), predShiftedPixelOrigin,
                                    maxDiff=self.tinyPixels)
        self.assertCoordsAlmostEqual(shiftedWcs.getSkyOrigin(), crval, maxDiff=self.tinyAngle)

        shiftedPixelList = [p + offset for p in pixelList]
        shiftedSkyList = shiftedWcs.applyForward(shiftedPixelList)
        self.assertCoordListsAlmostEqual(skyList, shiftedSkyList, maxDiff=self.tinyAngle)

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


class MetadataWcsTestCase(TransformTestBaseClass):
    """Test metadata constructor of SkyWcs
    """

    def setUp(self):
        self.metadata = PropertyList()

        self.metadata.set("SIMPLE", "T")
        self.metadata.set("BITPIX", -32)
        self.metadata.set("NAXIS", 2)
        self.metadata.set("NAXIS1", 1024)
        self.metadata.set("NAXIS2", 1153)
        self.metadata.set("RADESYS", "ICRS")
        self.metadata.set("EQUINOX", 2000.)

        self.metadata.setDouble("CRVAL1", 215.604025685476)
        self.metadata.setDouble("CRVAL2", 53.1595451514076)
        self.metadata.setDouble("CRPIX1", 1109.99981456774)
        self.metadata.setDouble("CRPIX2", 560.018167811613)
        self.metadata.set("CTYPE1", "RA---TAN")
        self.metadata.set("CTYPE2", "DEC--TAN")
        self.metadata.set("CUNIT1", "deg")
        self.metadata.set("CUNIT2", "deg")

        self.metadata.setDouble("CD1_1", 5.10808596133527E-05)
        self.metadata.setDouble("CD1_2", 1.85579539217196E-07)
        self.metadata.setDouble("CD2_2", -5.10281493481982E-05)
        self.metadata.setDouble("CD2_1", -1.85579539217196E-07)

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

    def testBasics(self):
        skyWcs = SkyWcs(self.metadata, strip = False)
        self.checkWcs(skyWcs)

    def testReadDESHeader(self):
        """Verify that we can read a DES header"""
        self.metadata.set("RADESYS", "ICRS   ")  # note trailing white space
        self.metadata.set("CTYPE1", "RA---TPV")
        self.metadata.set("CTYPE2", "DEC--TPV")

        skyWcs = SkyWcs(self.metadata, strip = False)
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

        wcs = SkyWcs(md, strip = False)

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

            wcs2 = SkyWcs(md, strip = False)
            skyPos2 = wcs2.pixelToSky(pixPos)
            self.assertCoordsAlmostEqual(skyPos2, pred_skyPos)


class WcsPairTransformTestCase(TransformTestBaseClass):
    """Test functionality of makeWcsPairTransform.
    """
    def setUp(self):
        TransformTestBaseClass.setUp(self)
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
                self.wcsList.append(SkyWcs(
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


class GetApproximateFitsWcsTestCase(TransformTestBaseClass):
    def setUp(self):
        self.dataPath = os.path.join(os.path.dirname(__file__), "data")

    def testTanSip(self):
        filePath = os.path.join(self.dataPath, "HSC-0908120-056-small.fits")
        # TODO DM-10765 read this using standard afw code once Wcs replaced with SkyWcs
        hdr = fits.getheader(filePath, 1)
        metadata = PropertyList()
        for name, value in hdr.items():
            metadata.set(name, value)

        wcs = SkyWcs(metadata)
        localTanWcs = wcs.getTanWcs(wcs.getPixelOrigin())
        bbox = Box2I(Point2I(0, 0), Extent2I(1000, 1000))  # arbitrary
        self.assertFalse(wcsAlmostEqualOverBBox(wcs, localTanWcs, bbox))

        approxMetadata = getApproximateFitsWcsMetadata(wcs)
        reconstitutedWcs = SkyWcs(approxMetadata)
        self.assertWcsAlmostEqualOverBBox(localTanWcs, reconstitutedWcs, bbox)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
