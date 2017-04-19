from __future__ import absolute_import, division, print_function
import unittest

import lsst.utils.tests
from lsst.afw.geom import SkyWcs, SpherePoint, Extent2D, Point2D, degrees, arcseconds, makeCdMatrix
from lsst.afw.geom.testUtils import TransformTestBaseClass


class TanSkyWcsTestCase(TransformTestBaseClass):
    """Test pure TAN constructor of SkyWcs
    """

    def setUp(self):
        TransformTestBaseClass.setUp(self)
        self.crpix = Point2D(100, 100)
        self.crval = SpherePoint(30 * degrees, 45 * degrees)
        self.scale = 1.0 * arcseconds

    def checkTanWcs(self, orientation, flipX):
        """Construct a pure TAN SkyWcs and check that it operates as specified

        Parameters
        ----------
        orientation : lsst.afw.geom.Angle
            Position angle of focal plane +Y, measured from N through E.
            At 0 degrees, +Y is along N and +X is along E/W if flipX false/true
            At 90 degrees, +Y is along E and +X is along S/N if flipX false/true
        flipX : bool
            Fip x axis? See `orientation` for details.

        Returns
        -------
        wcs : SkyWcs
            The generated pure TAN SkyWcs
        """
        wcs = SkyWcs(
            crpix = self.crpix,
            crval = self.crval,
            cdMatrix = makeCdMatrix(scale=self.scale, orientation=orientation, flipX=flipX),
        )
        self.checkPersistence(wcs)

        xoffAng = 180*degrees if flipX else 0*degrees

        pixelList = [
            Point2D(self.crpix[0], self.crpix[1]),
            Point2D(self.crpix[0] + 1, self.crpix[1]),
            Point2D(self.crpix[0], self.crpix[1] + 1),
        ]
        skyList = wcs.tranForward(pixelList)

        # check pixels to sky
        predSkyList = [
            self.crval,
            self.crval.offset(xoffAng - orientation, self.scale),
            self.crval.offset(90*degrees - orientation, self.scale)
        ]
        self.assertSpherePointListsAlmostEqual(predSkyList, skyList)
        self.assertSpherePointListsAlmostEqual(predSkyList, wcs.pixelToSky(pixelList))
        for pixel, predSky in zip(pixelList, predSkyList):
            self.assertSpherePointsAlmostEqual(predSky, wcs.pixelToSky(pixel))
            anglePair = wcs.pixelToSky(*pixel)
            self.assertSpherePointsAlmostEqual(predSky, SpherePoint(*anglePair))

        # check sky to pixels
        self.assertPairListsAlmostEqual(pixelList, wcs.tranInverse(skyList))
        self.assertPairListsAlmostEqual(pixelList, wcs.tranInverse(skyList))
        for pixel, sky in zip(pixelList, skyList):
            self.assertPairsAlmostEqual(pixel, wcs.skyToPixel(sky))
            xyPair = wcs.skyToPixel(*sky)
            self.assertPairsAlmostEqual(pixel, Point2D(*xyPair))

        crval = wcs.getSkyOrigin()
        self.assertSpherePointsAlmostEqual(crval, self.crval)

        crpix = wcs.getPixelOrigin()
        self.assertPairsAlmostEqual(crpix, self.crpix)

        cdMatrix = wcs.getCdMatrix()
        predCdMatrix = makeCdMatrix(scale=self.scale, orientation=orientation, flipX=flipX)
        self.assertFloatsAlmostEqual(cdMatrix, predCdMatrix)

        pixelScale = wcs.getPixelScale(self.crpix)
        self.assertAnglesAlmostEqual(self.scale, pixelScale)

        # Compute a WCS with the pixel origin shifted by an arbitrary amount
        # The resulting sky origin should not change
        offset = Extent2D(500, -322)  # arbitrary
        shiftedWcs = wcs.shiftedPixelOrigin(*offset)
        predShiftedPixelOrigin = self.crpix + offset
        self.assertPairsAlmostEqual(shiftedWcs.getPixelOrigin(), predShiftedPixelOrigin)
        self.assertSpherePointsAlmostEqual(shiftedWcs.getSkyOrigin(), self.crval)

        shiftedPixelList = [p + offset for p in pixelList]
        shiftedSkyList = shiftedWcs.tranForward(shiftedPixelList)
        self.assertSpherePointListsAlmostEqual(skyList, shiftedSkyList)

        return wcs

    def testOrient0FlipXFalse(self):
        self.checkTanWcs(orientation=0*degrees, flipX=False)

    def testOrient0FlipXTrue(self):
        self.checkTanWcs(orientation=0*degrees, flipX=True)

    def testOrient20FlipXFalse(self):
        self.checkTanWcs(orientation=20*degrees, flipX=False)

    def testOrient20FlipXTrue(self):
        self.checkTanWcs(orientation=20*degrees, flipX=True)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
