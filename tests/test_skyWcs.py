from __future__ import absolute_import, division, print_function
import itertools
import sys
import unittest

import lsst.afw.coord   # needed for assertCoordsNearlyEqual
import lsst.utils.tests
from lsst.afw.coord import IcrsCoord
from lsst.afw.geom import SkyWcs, Extent2D, Point2D, degrees, \
    arcseconds, radians, makeCdMatrix, makeWcsPairTransform
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
                    self.assertPairsNearlyEqual(
                        transform.applyInverse(point2),
                        point1)
                    self.assertCoordsNearlyEqual(
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
                self.assertPairsNearlyEqual(point, outPoint1)
                self.assertPairsNearlyEqual(outPoint1, outPoint2)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
