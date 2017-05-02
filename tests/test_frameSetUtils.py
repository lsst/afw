from __future__ import absolute_import, division, print_function
import unittest

from lsst.afw.geom import arcseconds, degrees, makeCdMatrix, Point2D, SpherePoint
from lsst.afw.geom.detail import makeTanWcsMetadata, readFitsWcs, readLsstSkyWcs
import lsst.utils.tests

PrintStrippedNames = False


class FrameSetUtilsTestCase(lsst.utils.tests.TestCase):
    """This is sparse because SkyWcs unit tests test much of this package
    """

    def setUp(self):
        # arbitrary values
        self.crpix = Point2D(100, 100)
        self.crval = SpherePoint(30 * degrees, 45 * degrees)
        self.scale = 1.0 * arcseconds

    def makeMetadata(self):
        """Return a WCS that is typical for an image

        It will contain 32 cards:
        - 14 standard WCS cards
        - 15 standard cards:
            - SIMPLE, BITPIX, NAXIS, NAXIS1, NAXIS2, BZERO, BSCALE
            - DATE-OBS, MJD-OBS, TIMESYS
            - EXPTIME
            - 2 COMMENT cards
            - INHERIT
            - EXTEND
        - LTV1 and LTV2, an IRAF convention LSST uses for image XY0
        - 1 nonstandard card
        """
        # arbitrary values
        orientation = 0 * degrees
        flipX = False
        metadata = makeTanWcsMetadata(
            crpix = self.crpix,
            crval = self.crval,
            cdMatrix = makeCdMatrix(scale=self.scale, orientation=orientation, flipX=flipX),
        )
        self.assertEqual(metadata.nameCount(), 14)
        metadata.add("SIMPLE", True)
        metadata.add("BITPIX", 16)
        metadata.add("NAXIS", 2)
        metadata.add("NAXIS1", 500)
        metadata.add("NAXIS2", 200)
        metadata.add("BZERO", 32768)
        metadata.add("BSCALE", 1)
        metadata.add("TIMESYS", "UTC")
        metadata.add("UTC-OBS", "12:04:45.73")
        metadata.add("DATE-OBS", "2006-05-20")
        metadata.add("EXPTIME", 5.0)
        metadata.add("COMMENT", "a comment")
        metadata.add("COMMENT", "another comment")
        metadata.add("EXTEND", True)
        metadata.add("INHERIT", False)
        metadata.add("LTV1", 5)
        metadata.add("LTV2", -10)
        metadata.add("ZOTHER", "non-standard")
        return metadata

    def testReadFitsWcsStripMetadata(self):
        metadata = self.makeMetadata()
        self.assertEqual(len(metadata.toList()), 32)
        readFitsWcs(metadata, strip=False)
        self.assertEqual(len(metadata.toList()), 32)
        readFitsWcs(metadata, strip=True)
        self.assertEqual(len(metadata.toList()), 18)

    def testReadLsstSkyWcsStripMetadata(self):
        metadata = self.makeMetadata()
        self.assertEqual(len(metadata.toList()), 32)
        readLsstSkyWcs(metadata, strip=False)
        self.assertEqual(len(metadata.toList()), 32)
        readLsstSkyWcs(metadata, strip=True)
        self.assertEqual(len(metadata.toList()), 18)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
