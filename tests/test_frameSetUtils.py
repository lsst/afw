from __future__ import absolute_import, division, print_function
import unittest

from astropy.coordinates import SkyCoord

import astshim as ast
from lsst.afw.geom import arcseconds, degrees, radians, Point2D, SpherePoint, makeCdMatrix
from lsst.afw.geom.detail import readFitsWcs, readLsstSkyWcs, getPropertyListFromFitsChan
from lsst.afw.geom.wcsUtils import makeSimpleWcsMetadata
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

        It will contain 14 cards that describe a WCS, added by makeSimpleWcsMetadata,
        plus 25 additional cards, including a set for WCS "A", which is what
        LSST uses to store xy0, and LTV1 and LTV2, which LSST used to use for xy0
        """
        # arbitrary values
        orientation = 0 * degrees
        flipX = False
        metadata = makeSimpleWcsMetadata(
            crpix = self.crpix,
            crval = self.crval,
            cdMatrix = makeCdMatrix(scale=self.scale, orientation=orientation, flipX=flipX),
        )
        self.assertEqual(metadata.nameCount(), 12)  # 2 CD terms are zero and so are omitted
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

    def getCrpix(self, metadata):
        """Get CRPIX from metadata using the LSST convention: 0-based in parent coordinates
        """
        return Point2D(   # zero-based, hence the - 1
            metadata.get("CRPIX1") + metadata.get("CRVAL1A") - 1,
            metadata.get("CRPIX2") + metadata.get("CRVAL2A") - 1,
        )

    def testReadFitsWcsStripMetadata(self):
        metadata = self.makeMetadata()
        nKeys = len(metadata.toList())
        nToStrip = 12
        frameSet1 = readFitsWcs(metadata, strip=False)
        self.assertEqual(type(frameSet1), ast.FrameSet)
        self.assertEqual(len(metadata.toList()), nKeys)

        # read again, this time stripping metadata
        frameSet2 = readFitsWcs(metadata, strip=True)
        self.assertEqual(len(metadata.toList()), nKeys - nToStrip)
        self.assertEqual(frameSet1, frameSet2)

        # having stripped the metadata, it should not be possible to create a WCS
        with self.assertRaises(lsst.pex.exceptions.TypeError):
            readFitsWcs(metadata, strip=False)

    def testReadFitsWcsFixRadecsys(self):
        # compare a WCS made with RADESYS against one made with RADECSYS,
        # using a system other than FK5, since that is the default;
        # both should be the same because readFitsWcs replaces RADECSYS with RADESYS
        metadata1 = self.makeMetadata()
        metadata1.set("RADESYS", "ICRS")
        frameSet1 = readFitsWcs(metadata1, strip=False)
        self.assertEqual(metadata1.get("RADESYS"), "ICRS")

        metadata2 = self.makeMetadata()
        metadata2.remove("RADESYS")
        metadata2.set("RADECSYS", "ICRS")
        frameSet2 = readFitsWcs(metadata2, strip=False)
        # metadata will have been corrected by readFitsWcs
        self.assertFalse(metadata2.exists("RADECSYS"))
        self.assertEqual(metadata2.get("RADESYS"), "ICRS")
        self.assertEqual(frameSet1, frameSet2)

    def testReadLsstSkyWcsStripMetadata(self):
        metadata = self.makeMetadata()
        nKeys = len(metadata.toList())
        nToStrip = 12 + 6  # WCS "A" is also stripped
        frameSet1 = readLsstSkyWcs(metadata, strip=False)
        self.assertEqual(len(metadata.toList()), nKeys)

        crval = SpherePoint(metadata.get("CRVAL1")*degrees, metadata.get("CRVAL2")*degrees)
        crvalRad = crval.getPosition(radians)
        desiredCrpix = self.getCrpix(metadata)
        computedCrpix = frameSet1.applyInverse(crvalRad)
        self.assertPairsAlmostEqual(desiredCrpix, computedCrpix)

        # read again, this time stripping metadata
        frameSet2 = readLsstSkyWcs(metadata, strip=True)
        self.assertEqual(len(metadata.toList()), nKeys - nToStrip)
        self.assertEqual(frameSet1, frameSet2)

        # having stripped the WCS keywords, we should not be able to generate
        # a WCS from what's left
        with self.assertRaises(lsst.pex.exceptions.TypeError):
            readLsstSkyWcs(metadata, strip=False)

        # try a full WCS with just CRPIX1 or 2 missing
        for i in (1, 2):
            metadata = self.makeMetadata()
            metadata.remove("CRPIX%d" % (i,))
            with self.assertRaises(lsst.pex.exceptions.TypeError):
                readLsstSkyWcs(metadata, strip=False)

    def testReadLsstSkyWcsNormalizeFk5(self):
        """Test that readLsstSkyWcs correctly normalizes FK5 1975 to ICRS
        """
        equinox = 1975
        metadata = self.makeMetadata()
        metadata.set("RADESYS", "FK5")
        metadata.set("EQUINOX", equinox)
        crpix = self.getCrpix(metadata)
        # record the original CRVAL before reading and stripping metadata
        crvalFk5Deg = (metadata.get("CRVAL1"), metadata.get("CRVAL2"))

        # read frameSet and compute crval
        frameSet = readLsstSkyWcs(metadata, strip=True)
        crvalRadians = frameSet.applyForward(crpix)
        crvalPoint = SpherePoint(*[crvalRadians[i]*radians for i in range(2)])

        # compare crval to the value recorded as the skyRef property of the current frame
        skyFrame = frameSet.getFrame(ast.FrameSet.CURRENT)
        recordedCrvalRadians = skyFrame.getSkyRef()
        recordedCrvalPoint = SpherePoint(*[recordedCrvalRadians[i]*radians for i in range(2)])
        self.assertSpherePointsAlmostEqual(crvalPoint, recordedCrvalPoint)

        # get predicted crval by converting with astropy
        crvalFk5 = SkyCoord(crvalFk5Deg[0], crvalFk5Deg[1], frame="fk5",
                            equinox="J%f" % (equinox,), unit="deg")
        predictedCrvalIcrs = crvalFk5.icrs
        predictedCrvalPoint = SpherePoint(predictedCrvalIcrs.ra.radian*radians,
                                          predictedCrvalIcrs.dec.radian*radians)
        # AST and astropy disagree by 0.025 arcsec; it's not worth worrying about because
        # we will always use ICRS internally and almost always fit our own WCS.
        self.assertSpherePointsAlmostEqual(crvalPoint, predictedCrvalPoint, maxSep=0.05*arcseconds)

    def testReadLsstSkyWcsNormalizeRaDec(self):
        """Test that a Dec, RA WCS frame set is normalized to RA, Dec
        """
        metadata = self.makeMetadata()

        crpix = self.getCrpix(metadata)

        # swap RA, Decaxes in metadata
        crvalIn = SpherePoint(metadata.get("CRVAL1")*degrees, metadata.get("CRVAL2")*degrees)
        metadata.set("CRVAL1", crvalIn[1].asDegrees())
        metadata.set("CRVAL2", crvalIn[0].asDegrees())
        metadata.set("CTYPE1", "DEC--TAN")
        metadata.set("CTYPE2", "RA---TAN")

        # create the wcs
        frameSet = readLsstSkyWcs(metadata)

        # compute pixel origin and compare to input crval
        computedCrvalRadians = frameSet.applyForward(crpix)
        computedCrval = SpherePoint(*[computedCrvalRadians[i]*radians for i in range(2)])
        self.assertSpherePointsAlmostEqual(crvalIn, computedCrval)

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

        # complex values are not supported by PropertyList
        fc.setFitsCF("UNSUPP", complex(1, 0))
        badCard = fc.getCard()
        with self.assertRaises(lsst.pex.exceptions.TypeError):
            getPropertyListFromFitsChan(fc)
        fc.setCard(badCard)
        fc.clearCard()
        fc.findFits("UNSUPP", inc=False)
        fc.delFits()
        metadata = getPropertyListFromFitsChan(fc)
        self.assertEqual(metadata.getOrderedNames(), expectedNames)

        # cards with no value are not supported by PropertyList
        fc.setFitsU("UNSUPP")
        badCard = fc.getCard()
        with self.assertRaises(lsst.pex.exceptions.TypeError):
            getPropertyListFromFitsChan(fc)
        fc.setCard(badCard)
        fc.clearCard()
        fc.findFits("UNSUPP", inc=False)
        fc.delFits()
        metadata = getPropertyListFromFitsChan(fc)
        self.assertEqual(metadata.getOrderedNames(), expectedNames)

    def testGetFitsCardsNoComments(self):
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
        with self.assertRaises(lsst.pex.exceptions.TypeError):
            getPropertyListFromFitsChan(fc)

        fc = ast.FitsChan(ast.StringStream())
        self.assertEqual(fc.className, "FitsChan")
        fc.setFitsU("UNDEFVAL")
        with self.assertRaises(lsst.pex.exceptions.TypeError):
            getPropertyListFromFitsChan(fc)

    def testMakeSimpleWcsMetadata(self):
        crpix = Point2D(111.1, 222.2)
        crval = SpherePoint(45.6 * degrees, 12.3 * degrees)
        scale = 1 * arcseconds
        for orientation in (0 * degrees, 21 * degrees):
            cdMatrix = makeCdMatrix(scale=scale, orientation=orientation)
            for projection in ("TAN", "STG"):
                metadata = makeSimpleWcsMetadata(crpix=crpix, crval=crval,
                                                 cdMatrix=cdMatrix, projection=projection)
                desiredLength = 12 if orientation == 0 * degrees else 14
                self.assertEqual(len(metadata.names()), desiredLength)
                self.assertEqual(metadata.get("RADESYS"), "ICRS")
                self.assertAlmostEqual(metadata.get("EQUINOX"), 2000.0)
                self.assertEqual(metadata.get("CTYPE1"), "RA---" + projection)
                self.assertEqual(metadata.get("CTYPE2"), "DEC--" + projection)
                for i in range(2):
                    self.assertAlmostEqual(metadata.get("CRPIX%d" % (i + 1,)), crpix[i] + 1)
                    self.assertAlmostEqual(metadata.get("CRVAL%d" % (i + 1,)), crval[i].asDegrees())
                    self.assertEqual(metadata.get("CUNIT%d" % (i + 1,)), "deg")
                for i in range(2):
                    for j in range(2):
                        name = "CD%d_%d" % (i + 1, j + 1)
                        if cdMatrix[i, j] != 0:
                            self.assertAlmostEqual(metadata.get(name), cdMatrix[i, j])
                        else:
                            self.assertFalse(metadata.exists(name))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
