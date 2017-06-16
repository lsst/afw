from __future__ import absolute_import, division, print_function
import itertools
import sys
import unittest

import lsst.utils.tests
import lsst.daf.base
from lsst.afw.geom import SkyWcs, SpherePoint, Extent2D, Point2D, degrees, \
    arcseconds, radians, makeCdMatrix
from lsst.afw.geom.testUtils import TransformTestBaseClass


class TanSkyWcsTestCase(TransformTestBaseClass):
    """Test pure TAN constructor of SkyWcs
    """

    def setUp(self):
        TransformTestBaseClass.setUp(self)
        self.crpix = Point2D(100, 100)
        self.crvalList = [
            SpherePoint(0 * degrees, 45 * degrees),
            SpherePoint(0.00001 * degrees, 45 * degrees),
            SpherePoint(359.99999 * degrees, 45 * degrees),
            SpherePoint(30 * degrees, 89.99999 * degrees),
            SpherePoint(30 * degrees, -89.99999 * degrees),
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
        crval : `lsst.afw.geom.Angle`
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
                return crval.offset(direction, amount)
            except Exception:
                return SpherePoint(crval[0] + direction, crval - amount)

        pixelList = [
            Point2D(self.crpix[0], self.crpix[1]),
            Point2D(self.crpix[0] + 1, self.crpix[1]),
            Point2D(self.crpix[0], self.crpix[1] + 1),
        ]
        skyList = wcs.tranForward(pixelList)

        # check pixels to sky
        predSkyList = [
            crval,
            safeOffset(crval, xoffAng - orientation, self.scale),
            safeOffset(crval, 90*degrees - orientation, self.scale),
        ]
        self.assertSpherePointListsAlmostEqual(predSkyList, skyList)
        self.assertSpherePointListsAlmostEqual(predSkyList, wcs.pixelToSky(pixelList))
        for pixel, predSky in zip(pixelList, predSkyList):
            self.assertSpherePointsAlmostEqual(predSky, wcs.pixelToSky(pixel))
            anglePair = wcs.pixelToSky(*pixel)
            self.assertSpherePointsAlmostEqual(predSky, SpherePoint(*anglePair))

        # check sky to pixels
        self.assertPairListsAlmostEqual(pixelList, wcs.applyInverse(skyList))
        self.assertPairListsAlmostEqual(pixelList, wcs.applyInverse(skyList))
        for pixel, sky in zip(pixelList, skyList):
            self.assertPairsAlmostEqual(pixel, wcs.skyToPixel(sky))
            xyPair = wcs.skyToPixel(*sky)
            self.assertPairsAlmostEqual(pixel, Point2D(*xyPair))

        crval = wcs.getSkyOrigin()
        self.assertSpherePointsAlmostEqual(crval, crval, maxSep=self.tinyAngle)

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
        self.assertSpherePointsAlmostEqual(shiftedWcs.getSkyOrigin(), crval, maxSep=self.tinyAngle)

        shiftedPixelList = [p + offset for p in pixelList]
        shiftedSkyList = shiftedWcs.tranForward(shiftedPixelList)
        self.assertSpherePointListsAlmostEqual(skyList, shiftedSkyList, maxSep=self.tinyAngle)

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
        self.metadata = lsst.daf.base.PropertySet()

        self.metadata.set("SIMPLE", "T")
        self.metadata.set("BITPIX", -32)
        self.metadata.set("NAXIS", 2)
        self.metadata.set("NAXIS1", 1024)
        self.metadata.set("NAXIS2", 1153)
        self.metadata.set("RADECSYS", 'FK5')
        self.metadata.set("EQUINOX", 2000.)

        self.metadata.setDouble("CRVAL1", 215.604025685476)
        self.metadata.setDouble("CRVAL2", 53.1595451514076)
        self.metadata.setDouble("CRPIX1", 1109.99981456774)
        self.metadata.setDouble("CRPIX2", 560.018167811613)
        self.metadata.set("CTYPE1", 'RA---SIN')
        self.metadata.set("CTYPE2", 'DEC--SIN')

        self.metadata.setDouble("CD1_1", 5.10808596133527E-05)
        self.metadata.setDouble("CD1_2", 1.85579539217196E-07)
        self.metadata.setDouble("CD2_2", -5.10281493481982E-05)
        self.metadata.setDouble("CD2_1", -8.27440751733828E-07)

    def checkWcs(self, skyWcs):
        cdMatrix = skyWcs.getCdMatrix()
        for i, j in itertools.izip(range(2), range(2)):
            self.assertAlmostEqual(cdMatrix[i, j], self.metadata.get("CD%s_%s" % (i+1, j+1)))
        pixelOrigin = skyWcs.getPixelOrigin()
        skyOrigin = skyWcs.getSkyOrigin()
        for i in range(2):
            self.assertAlmostEqual(pixelOrigin[i], self.metadata.get("CRPIX%s" % (i+1,)))
            self.assertAlmostEqual(skyOrigin[i], self.metadata.get("CRVAL%s" % (i+1,)))

    def testBasics(self):
        skyWcs = SkyWcs(self.metadata)
        self.checkWcs(skyWcs)

    def testReadDESHeader(self):
        """Verify that we can read a DES header"""
        self.metadata.set("RADESYS", "FK5    ")  # note trailing white space
        self.metadata.set("CTYPE1", 'RA---TPV')
        self.metadata.set("CTYPE2", 'DEC--TPV')

        skyWcs = SkyWcs(self.metadata)
        self.checkWcs(skyWcs)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
