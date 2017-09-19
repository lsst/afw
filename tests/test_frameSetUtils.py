from __future__ import absolute_import, division, print_function
import unittest

import astshim as ast
from lsst.afw.coord import IcrsCoord
from lsst.afw.geom import arcseconds, degrees, makeCdMatrix, Point2D
from lsst.afw.geom.detail import makeTanWcsMetadata, readFitsWcs, readLsstSkyWcs, getPropertyListFromFitsChan
import lsst.utils.tests

PrintStrippedNames = False


class FrameSetUtilsTestCase(lsst.utils.tests.TestCase):
    """This is sparse because SkyWcs unit tests test much of this package
    """

    def setUp(self):
        # arbitrary values
        self.crpix = Point2D(100, 100)
        self.crval = IcrsCoord(30 * degrees, 45 * degrees)
        self.scale = 1.0 * arcseconds

    def makeMetadata(self):
        """Return a WCS that is typical for an image

        It will contain 14 cards that describe a WCS, added by makeTanWcsMetadata,
        plus 25 additional cards, including a set for WCS "A", which is what
        LSST uses to store xy0, and LTV1 and LTV2, which LSST used to use for xy0
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
        metadata.add("HISTORY", "some history")
        metadata.add("EXTEND", True)
        metadata.add("INHERIT", False)
        metadata.add("CRPIX1A", 1.0)
        metadata.add("CRPIX2A", 1.0)
        metadata.add("CRVAL1A", 300)
        metadata.add("CRVAL2A", 400)
        metadata.add("CUNIT1A", "pixels")
        metadata.add("CUNIT2A", "pixels")
        metadata.add("LTV1", -300)
        metadata.add("LTV2", -400)
        metadata.add("ZOTHER", "non-standard")
        return metadata

    def testReadFitsWcsStripMetadata(self):
        metadata = self.makeMetadata()
        self.assertEqual(len(metadata.toList()), 39)
        readFitsWcs(metadata, strip=False)
        self.assertEqual(len(metadata.toList()), 39)
        readFitsWcs(metadata, strip=True)
        # this should strip the 14 items added by makeTanWcsMetadata
        self.assertEqual(len(metadata.toList()), 25)

    def testReadLsstSkyWcsStripMetadata(self):
        metadata = self.makeMetadata()
        self.assertEqual(len(metadata.toList()), 39)
        readLsstSkyWcs(metadata, strip=False)
        self.assertEqual(len(metadata.toList()), 39)
        readLsstSkyWcs(metadata, strip=True)
        self.assertEqual(len(metadata.toList()), 25)

    def testGetPropertyListFromFitsChanWithComments(self):
        fc = ast.FitsChan(ast.StringStream())
        self.assertEqual(fc.className, "FitsChan")

        # add one card for each supported type, with a comment
        continueVal = "This is a continue card"
        floatVal = 1.5
        intVal = 99
        logicalVal = True
        stringVal = "This is a string"
        fc.setFitsCN("ACONT", continueVal, "Comment for ACONT")
        fc.setFitsF("AFLOAT", floatVal, "Comment for AFLOAT")
        fc.setFitsI("ANINT", intVal, "Comment for ANINT")
        fc.setFitsL("ALOGICAL", logicalVal, "Comment for ALOGICAL")
        fc.setFitsS("ASTRING", stringVal, "Comment for ASTRING")
        fc.setFitsCM("a comment, which will be ignored by getPropertyListFromFitsChan")
        expectedNames = ["ACONT", "AFLOAT", "ANINT", "ALOGICAL", "ASTRING"]

        self.assertEqual(fc.nCard, 6)
        metadata = getPropertyListFromFitsChan(fc)
        self.assertEqual(metadata.getOrderedNames(), expectedNames)

        self.assertEqual(metadata.get("ACONT"), continueVal)
        self.assertAlmostEqual(metadata.get("AFLOAT"), floatVal)
        self.assertEqual(metadata.get("ANINT"), intVal)
        self.assertEqual(metadata.get("ALOGICAL"), logicalVal)
        self.assertEqual(metadata.get("ASTRING"), stringVal)
        self.assertEqual(metadata.get("ACONT"), continueVal)

        for name in expectedNames:
            self.assertEqual(metadata.getComment(name), "Comment for %s" % (name,))

    def testGtFitsCardsNoComments(self):
        fc = ast.FitsChan(ast.StringStream())
        self.assertEqual(fc.className, "FitsChan")

        # add one card for each supported type, with a comment
        continueVal = "This is a continue card"
        floatVal = 1.5
        intVal = 99
        logicalVal = True
        stringVal = "This is a string"
        fc.setFitsCN("ACONT", continueVal)
        fc.setFitsF("AFLOAT", floatVal)
        fc.setFitsI("ANINT", intVal)
        fc.setFitsL("ALOGICAL", logicalVal)
        fc.setFitsS("ASTRING", stringVal)
        fc.setFitsCM("a comment, which will be ignored by getPropertyListFromFitsChan")

        self.assertEqual(fc.nCard, 6)
        metadata = getPropertyListFromFitsChan(fc)
        self.assertEqual(metadata.getOrderedNames(), ["ACONT", "AFLOAT", "ANINT", "ALOGICAL", "ASTRING"])

        self.assertEqual(metadata.get("ACONT"), continueVal)
        self.assertAlmostEqual(metadata.get("AFLOAT"), floatVal)
        self.assertEqual(metadata.get("ANINT"), intVal)
        self.assertEqual(metadata.get("ALOGICAL"), logicalVal)
        self.assertEqual(metadata.get("ASTRING"), stringVal)
        self.assertEqual(metadata.get("ACONT"), continueVal)

    def testGetPropertyListFromFitsChanUnsupportedTypes(self):
        fc = ast.FitsChan(ast.StringStream())
        self.assertEqual(fc.className, "FitsChan")
        fc.setFitsCF("ACOMPLEX", complex(1, 1))
        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
            getPropertyListFromFitsChan(fc)

        fc = ast.FitsChan(ast.StringStream())
        self.assertEqual(fc.className, "FitsChan")
        fc.setFitsU("UNDEFVAL")
        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
            getPropertyListFromFitsChan(fc)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
