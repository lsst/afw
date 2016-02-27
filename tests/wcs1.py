#!/usr/bin/env python2
from __future__ import absolute_import, division

# 
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
# 
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and 
# the GNU General Public License along with this program.  If not, 
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import os.path
import math
import unittest

import lsst.utils
import lsst.daf.base as dafBase
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
import lsst.utils.tests as utilsTests
import lsst.afw.display.ds9 as ds9
import lsst
import numpy

try:
    type(verbose)
except NameError:
    verbose = 0

dataDir = lsst.utils.getPackageDir("afwdata")
if not dataDir:
    raise RuntimeError("Must set up afwdata to run these tests")
InputImagePath = os.path.join(dataDir, "data", "871034p_1_MI")
InputSmallImagePath = os.path.join(dataDir, "data", "small_img.fits")
InputCorruptMaskedImageName = "small_MI_corrupt"
currDir = os.path.abspath(os.path.dirname(__file__))
InputCorruptFilePath = os.path.join(currDir, "data", InputCorruptMaskedImageName)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

CoordSysList = [afwCoord.ICRS, afwCoord.FK5, afwCoord.ECLIPTIC, afwCoord.GALACTIC]

def coordSysEquinoxIter(equinoxList=(1950, 1977, 2000)):
    for coordSys in CoordSysList:
        if coordSys in (afwCoord.FK5, afwCoord.ECLIPTIC):
            for equinox in equinoxList:
                yield (coordSys, equinox)
        else:
            yield (coordSys, 2000) # Coord's default

def makeWcs(
    pixelScale=0.2*afwGeom.arcseconds,
    crPixPos=(2000.0, 1520.5),
    crValDeg=(27.53, 87.123),
    posAng=afwGeom.Angle(0.0),
    doFlipX=False,
    projection="TAN",
    coordSys=afwCoord.ICRS,
    equinox=2000
):
    """Make an simple TAN WCS with sensible defaults
    
    @param[in] pixelScale: desired scale, as sky/pixel, an afwGeom.Angle
    @param[in] crPixPos: crPix for WCS, using the LSST standard; a pair of floats
    @param[in] crValDeg: crVal for WCS, in degrees; a pair of floats
    @param[in] posAng: position angle (afwGeom.Angle)
    @param[in] doFlipX: flip X axis?
    @param[in] projection: WCS projection (e.g. "TAN" or "STG")
    @param[in] coordSys: coordinate system enum
    @param[in] equinox: date of equinox; should be 2000 for ICRS or GALACTIC
    """
    if len(projection) != 3:
        raise RuntimeError("projection=%r; must have length 3" % (projection,))

    csysPrefixes, radesys = {
        afwCoord.ICRS: (("RA", "DEC"), "ICRS"),
        afwCoord.FK5:  (("RA", "DEC"), "FK5"),
        afwCoord.ECLIPTIC: (("ELON", "ELAT"), None),
        afwCoord.GALACTIC: (("GLON", "GLAT"), None),
    }[coordSys]

    ctypeList = [("%-4s-%3s" % (csysPrefixes[i], projection)).replace(" ", "-") for i in range(2)]
    ps = dafBase.PropertySet()
    crPixFits = [ind + 1.0 for ind in crPixPos] # convert pix position to FITS standard
    posAngRad = posAng.asRadians()
    pixelScaleDeg = pixelScale.asDegrees()
    cdMat = numpy.array([[ math.cos(posAngRad), math.sin(posAngRad)],
                         [-math.sin(posAngRad), math.cos(posAngRad)]], dtype=float) * pixelScaleDeg
    if doFlipX:
        cdMat[:,0] = -cdMat[:,0]
    for i in range(2):
        ip1 = i + 1
        ps.add("CTYPE%1d" % (ip1,), ctypeList[i])
        ps.add("CRPIX%1d" % (ip1,), crPixFits[i])
        ps.add("CRVAL%1d" % (ip1,), crValDeg[i])
    if radesys:
        ps.add("RADESYS", radesys)
    ps.add("EQUINOX", equinox)
    ps.add("CD1_1", cdMat[0, 0])
    ps.add("CD2_1", cdMat[1, 0])
    ps.add("CD1_2", cdMat[0, 1])
    ps.add("CD2_2", cdMat[1, 1])
    return afwImage.makeWcs(ps)

def localMakeCoord(coordSys, posDeg, equinox):
    """Make a coord, ignoring equinox if necessary
    """
    if coordSys in (afwCoord.ICRS, afwCoord.GALACTIC):
        return afwCoord.makeCoord(coordSys, posDeg[0]*afwGeom.degrees, posDeg[1]*afwGeom.degrees)
    return afwCoord.makeCoord(coordSys, posDeg[0]*afwGeom.degrees, posDeg[1]*afwGeom.degrees, equinox)

class WcsTestCase(utilsTests.TestCase):
    def testCD_PC(self):
        """Test that we can read a FITS file with both CD and PC keys (like early Suprimecam files)"""
        md = dafBase.PropertyList()
        for k, v in (
            ("EQUINOX", 2000.0),
            ("RADESYS", 'FK5'),
            ("CRPIX1" , 5353.0),
            ("CRPIX2" , -35.0),
            ("CD1_1"  , 0.0),
            ("CD1_2"  , -5.611E-05),
            ("CD2_1"  , -5.611E-05),
            ("CD2_2"  , -0.0),
            ("CRVAL1" , 4.5789875),
            ("CRVAL2" , 16.30004444),
            ("CUNIT1" , 'deg'),
            ("CUNIT2" , 'deg'),
            ("CTYPE1" , 'RA---TAN'),
            ("CTYPE2" , 'DEC--TAN'),
            ("CDELT1" , -5.611E-05),
            ("CDELT2" , 5.611E-05),
            ):
            md.set(k, v)

        wcs = afwImage.makeWcs(md)

        x, y = 1000, 2000
        ra, dec = 4.459815023498577, 16.544199850984768

        sky = wcs.pixelToSky(x, y)
        for i, v in enumerate([ra, dec]):
            self.assertEqual(sky[i].asDegrees(), v)

        for badPC in (False, True):
            if verbose:
                print "Checking PC coefficients: badPC =", badPC
            for k, v in (
                ("PC001001",  0.0),
                ("PC001002", -1.0 if badPC else 1.0),
                ("PC002001",  1.0 if badPC else -1.0),
                ("PC002002",  0.0),
                ):
                md.set(k, v)

            # Check Greisen and Calabretta A&A 395 1061 (2002), Eq. 3
            if not badPC:
                for i in (1, 2,):
                    for j in (1, 2,):
                        self.assertEqual(md.get("CD%d_%d" % (i, j)), 
                                         md.get("CDELT%d" % i)*md.get("PC00%d00%d" % (i, j)))

            wcs = afwImage.makeWcs(md)
            sky = wcs.pixelToSky(x, y)
            for i, v in enumerate([ra, dec]):
                self.assertEqual(sky[i].asDegrees(), v)

    def testCast(self):
        # strangely, this overload of makeWcs returns a TAN WCS that's not a TanWcs
        wcs = afwImage.makeWcs(afwCoord.IcrsCoord(45.0*afwGeom.degrees, 45.0*afwGeom.degrees),
                               afwGeom.Point2D(0.0, 0.0), 1.0, 0.0, 0.0, 1.0)
        # ...but if you round-trip it through a PropertySet, you get a TanWcs in a Wcs ptr,
        # which is what we want for this test.
        base = afwImage.makeWcs(wcs.getFitsMetadata())
        self.assertEqual(type(base), afwImage.Wcs)
        derived = afwImage.TanWcs.cast(base)
        self.assertEqual(type(derived), afwImage.TanWcs)

    def testCoordConversion(self):
        """Test that skyToPixel and pixelToSky handle coordinate system and equinox correctly

        Given two WCS that are identical except perhaps for coordinate system and equinox
        compute: sky2 = wcs2.pixelToSky(wcs1.skyToPixel(sky1)
        The result should be the same numerical values, but with wcs2's coordinate system and equinox
        """
        crValDeg=(27.53, 87.123)
        for coordSys1, equinox1 in coordSysEquinoxIter():
            wcs1 = makeWcs(crValDeg=crValDeg, coordSys=coordSys1, equinox=equinox1)
            for coordSys2, equinox2 in coordSysEquinoxIter():
                wcs2 = makeWcs(crValDeg=crValDeg, coordSys=coordSys2, equinox=equinox2)
                for sky1Deg in (crValDeg, (18.52, 46.765)):
                    coord1 = localMakeCoord(coordSys1, sky1Deg, equinox1)
                    pixPos = wcs1.skyToPixel(coord1)
                    coord2 = wcs2.pixelToSky(pixPos)

                    desCoord2 = localMakeCoord(coordSys2, (coord1[0].asDegrees(), coord1[1].asDegrees()), equinox2)
                    self.assertCoordsNearlyEqual(coord2, desCoord2)

    def testGetCoordSys(self):
        """Test getCoordSystem, getEquinox"""
        def isIcrs(wcs):
            """Return True if wcs is ICRS or FK5 J2000"""
            csys = wcs.getCoordSystem()
            if csys == afwCoord.ICRS:
                return True
            return csys == afwCoord.FK5 and wcs.getEquinox() == 2000

        def refIsSameSkySystem(wcs1, wcs2):
            if isIcrs(wcs1) and isIcrs(wcs2):
                return True
            return (wcs1.getCoordSystem() == wcs2.getCoordSystem()) and (wcs1.getEquinox() == wcs2.getEquinox())

        for coordSys, equinox in coordSysEquinoxIter():
            wcs = makeWcs(coordSys=coordSys, equinox=equinox)
            self.assertEqual(wcs.getCoordSystem(), coordSys)
            if coordSys not in (afwCoord.ICRS, afwCoord.GALACTIC):
                # ICRS and Galactic by definition does not have an equinox.
                self.assertEqual(wcs.getEquinox(), equinox)
            predIsIcrs = coordSys == afwCoord.ICRS or (coordSys == afwCoord.FK5 and equinox == 2000)
            self.assertEqual(predIsIcrs, isIcrs(wcs))
            for coordSys2, equinox2 in coordSysEquinoxIter():
                wcs2 = makeWcs(coordSys=coordSys2, equinox=equinox2)
                self.assertEqual(refIsSameSkySystem(wcs, wcs2), wcs.isSameSkySystem(wcs2))

    def testIcrsEquinox(self):
        """Check that EQUINOX is not written to FITS for ICRS coordinates"""
        def checkEquinoxHeader(coordSysName, writeEquinox):
            coordSys = getattr(afwCoord, coordSysName)
            # We should get the same behaviour with both Wcs and TanWcs: check them both.
            for dummyWcs in (makeWcs(coordSys=coordSys), afwImage.TanWcs.cast(makeWcs(coordSys=coordSys))):
                dummyExposure = afwImage.ExposureF()
                dummyExposure.setWcs(dummyWcs)
                with utilsTests.getTempFilePath(".fits") as tmpFile:
                    dummyExposure.writeFits(tmpFile)
                    metadata = afwImage.readMetadata(tmpFile)
                    self.assertEqual(metadata.get("RADESYS"), coordSysName)
                    self.assertTrue(("EQUINOX" in metadata.names()) == writeEquinox)
        checkEquinoxHeader("ICRS", False)
        checkEquinoxHeader("FK5", True)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class WCSRotateFlip(unittest.TestCase):
    """A test case for the methods to rotate and flip a wcs under similar operations to the image pixels"""
    def setUp(self):
        self.size = afwGeom.Extent2I(509, 2000)

    def tearDown(self):
        del self.size

    def makeWcs(self):
        crval = afwCoord.Coord(afwGeom.Point2D(1.606631, 5.090329))
        crpix = afwGeom.Point2D(2036., 2000.)
        return afwImage.makeWcs(crval, crpix, 5.399452e-5, -1.30770e-5, 1.30770e-5, 5.399452e-5)

    def testRotation(self):
        # Origin for LSST pixels is (0,0).  Need to subtract one when rotating to avoid off by one.
        # E.g. UR (507, 1999) goes to (0,0) for nRot = 2
        q1 = {0:afwGeom.Point2D(100., 1600.), 
              1:afwGeom.Point2D(self.size.getY() - 1600. - 1, 100.),
              2:afwGeom.Point2D(self.size.getX() - 100. - 1, self.size.getY() - 1600. - 1),
              3:afwGeom.Point2D(1600., self.size.getX() - 100. - 1)}
        wcs = self.makeWcs()
        pos0 = wcs.pixelToSky(q1[0])
        for rot in (1,2,3):
            wcs = self.makeWcs()
            wcs.rotateImageBy90(rot, self.size)
            self.assertEqual(pos0, wcs.pixelToSky(q1[rot]))

    def testFlip(self):
        q1 = {'noFlip': afwGeom.Point2D(300., 900.),
              'flipLR': afwGeom.Point2D(self.size.getX()-300.-1, 900.),
              'flipTB': afwGeom.Point2D(300., self.size.getY()-900.-1)}
        wcs = self.makeWcs()
        pos0 = wcs.pixelToSky(q1['noFlip'])
        wcs = self.makeWcs()
        wcs.flipImage(True, False, self.size)
        self.assertEqual(pos0, wcs.pixelToSky(q1['flipLR']))
        wcs = self.makeWcs()
        wcs.flipImage(False, True, self.size)
        self.assertEqual(pos0, wcs.pixelToSky(q1['flipTB']))

class WCSTestCaseSDSS(unittest.TestCase):
    """A test case for WCS using a small (SDSS) image with a slightly weird WCS"""

    def setUp(self):
        self.im = afwImage.DecoratedImageD(InputSmallImagePath)

        self.wcs = afwImage.makeWcs(self.im.getMetadata())

        if False:
            ds9.mtv(self.im, wcs=self.wcs)

    def tearDown(self):
        del self.wcs
        del self.im

    def testCrpix(self):
        metadata = self.im.getMetadata()
        crpix0 = metadata.getAsDouble("CRPIX1")
        crpix1 = metadata.getAsDouble("CRPIX2")

        lsstCrpix = self.wcs.getPixelOrigin()

        self.assertEqual(lsstCrpix[0], crpix0-1)
        self.assertEqual(lsstCrpix[1], crpix1-1)

    def testXyToRaDecArguments(self):
        """Check that conversion of xy to ra dec (and back again) works"""
        xy = afwGeom.Point2D(110, 123)
        raDec = self.wcs.pixelToSky(xy)
        xy2 = self.wcs.skyToPixel(raDec)

        self.assertAlmostEqual(xy.getX(), xy2.getX())
        self.assertAlmostEqual(xy.getY(), xy2.getY())

        raDec = afwCoord.makeCoord(afwCoord.ICRS, 245.167400 * afwGeom.degrees, +19.1976583 * afwGeom.degrees)

        xy = self.wcs.skyToPixel(raDec)
        raDec2 = self.wcs.pixelToSky(xy)

        self.assertAlmostEqual(raDec[0].asDegrees(), raDec2[0].asDegrees())
        self.assertAlmostEqual(raDec[1].asDegrees(), raDec2[1].asDegrees())

    def test_RaTan_DecTan(self):
        """Check the RA---TAN, DEC--TAN WCS conversion"""
        # values from wcstools xy2sky (v3.8.1). Confirmed by ds9
        raDec0 = afwGeom.Point2D(245.15984167, +19.1960472) 
        raDec = self.wcs.pixelToSky(0.0, 0.0).getPosition()

        self.assertAlmostEqual(raDec.getX(), raDec0.getX(), 5)
        self.assertAlmostEqual(raDec.getY(), raDec0.getY(), 5) 

    def testIdentity(self):
        """Convert from ra, dec to col, row and back again"""
        raDec = afwCoord.makeCoord(afwCoord.ICRS, 244 * afwGeom.degrees, 20 * afwGeom.degrees)
        if verbose:
            print 'testIdentity'
            print 'wcs:'
            for x in self.wcs.getFitsMetadata().toList():
                print '  ', x
            print 'raDec:', raDec
            print type(self.wcs)
        rowCol = self.wcs.skyToPixel(raDec)
        raDec2 = self.wcs.pixelToSky(rowCol)

        if verbose:
            print 'rowCol:', rowCol
            print 'raDec2:', raDec2

        p1 = raDec.getPosition()
        p2 = raDec.getPosition()
        if verbose:
            print 'p1,p2', p1,p2
        self.assertAlmostEqual(p1[0], p2[0])
        self.assertAlmostEqual(p1[1], p2[1])

    def testInvalidRaDec(self):
        """Test a conversion for an invalid position.  Well, "test" isn't
        quite right as the result is invalid, but make sure that it still is"""
        raDec = afwCoord.makeCoord(afwCoord.ICRS, 1 * afwGeom.degrees, 2 * afwGeom.degrees)

        self.assertRaises(lsst.pex.exceptions.Exception, self.wcs.skyToPixel, raDec)

    def testCD(self):
        self.wcs.getCDMatrix()

    def testCreateCDMatrix(self):
        """Test that we make a correct CD matrix even if the header only has a PC matrix"""
        md = dafBase.PropertySet()
        md.set("NAXIS", 2)
        md.set("CTYPE1", "RA---TAN")
        md.set("CTYPE2", "DEC--TAN")
        md.set("CRPIX1", 0)
        md.set("CRPIX2", 0)
        md.set("CRVAL1", 0)
        md.set("CRVAL2", 0)
        md.set("RADECSYS", "FK5")
        md.set("EQUINOX", 2000.0)

        wcs = afwImage.makeWcs(md)
        self.assertTrue(numpy.all(wcs.getCDMatrix() == numpy.array([[1.0, 0.0], [0.0, 1.0]])))

        md.set("PC1_1", 2)
        wcs = afwImage.makeWcs(md)
        self.assertTrue(numpy.all(wcs.getCDMatrix() == numpy.array([[2.0, 0.0], [0.0, 1.0]])))

    def testStripKeywords(self):
        """Test that we can strip WCS keywords from metadata when constructing a Wcs"""
        metadata = self.im.getMetadata()
        self.wcs = afwImage.makeWcs(metadata)

        self.assertTrue(metadata.exists("CRPIX1"))

        strip = True
        self.wcs = afwImage.makeWcs(metadata, strip)
        self.assertFalse(metadata.exists("CRPIX1"))

    def testAffineTransform(self):
        self.wcs.getLinearTransform()
        self.wcs.getCDMatrix()

    def testXY0(self):
        """Test that XY0 values are handled correctly when building an exposure and also when
        reading the WCS header directly.  #2205"""
        bbox = afwGeom.Box2I(afwGeom.Point2I(1000, 1000), afwGeom.Extent2I(10, 10))

        def makeWcs(crPixPos, crValDeg, projection):
            ps = dafBase.PropertySet()
            ctypes = [("%-5s%3s" % (("RA", "DEC")[i], projection)).replace(" ", "-") for i in range(2)]
            for i in range(2):
                ip1 = i + 1
                ps.add("CTYPE%1d" % (ip1,), ctypes[i])
                ps.add("CRPIX%1d" % (ip1,), crPixPos[i])
                ps.add("CRVAL%1d" % (ip1,), crValDeg[i])
            ps.add("RADECSYS", "ICRS")
            ps.add("EQUINOX", 2000)
            ps.add("CD1_1", -0.001)
            ps.add("CD2_1", 0.0)
            ps.add("CD1_2", 0.0)
            ps.add("CD2_2", 0.001)
            return afwImage.makeWcs(ps)

        wcs = makeWcs(
            crPixPos = (100.0, 100.0),
            crValDeg = (10.0, 35.0),
            projection = "STG", # also fails for TAN
        )

        exposure = afwImage.ExposureF(bbox, wcs)
        pixPos = afwGeom.Box2D(bbox).getMax()
        if verbose:
            print "XY0=", exposure.getXY0()
            print "pixPos=", pixPos
        skyPos = wcs.pixelToSky(pixPos)

        with utilsTests.getTempFilePath(".fits") as tmpFile:
            exposure.writeFits(tmpFile)
            for useExposure in (False, True):
                if useExposure:
                    unpExp = afwImage.ExposureF(tmpFile)
                    unpWcs = unpExp.getWcs()
                else:
                    md = afwImage.readMetadata(tmpFile)
                    unpWcs = afwImage.makeWcs(md, False)
                unpPixPos = unpWcs.skyToPixel(skyPos)

                if verbose:
                    print "useExposure=%s; unp pixPos=%s" % (useExposure, unpPixPos)

                for i in range(2):
                    self.assertAlmostEqual(unpPixPos[i], 1009.5)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class WCSTestCaseCFHT(unittest.TestCase):
    """A test case for WCS"""

    def setUp(self):
        path = InputImagePath + ".fits"
        self.metadata = afwImage.readMetadata(path)
        self.wcs = afwImage.makeWcs(self.metadata)

    def tearDown(self):
        del self.wcs
        del self.metadata

    def test_RaTan_DecTan(self):
        """Check the RA---TAN, DEC--TAN WCS conversion"""
        raDec = self.wcs.pixelToSky(0.0, 0.0).getPosition() # position read off ds9

        self.assertAlmostEqual(raDec[0], 17.87673, 5) # ra from ds9
        self.assertAlmostEqual(raDec[1],  7.72231, 5) # dec from ds9

    def testPlateScale(self):
        """Test that we can measure the area of a pixel"""

        p00 = afwGeom.Point2D(10, 10)
        p00 = afwGeom.Point2D(self.metadata.getAsDouble("CRPIX1"), self.metadata.getAsDouble("CRPIX2"))

        sky00 = self.wcs.pixelToSky(p00)
        cosdec = math.cos(sky00[1])

        side = 1e-3
        icrs = afwCoord.ICRS
        sky10 = afwCoord.makeCoord(
            icrs, sky00.getPosition(afwGeom.degrees) + afwGeom.Extent2D(side/cosdec, 0), afwGeom.degrees
        )
        sky01 = afwCoord.makeCoord(
            icrs, sky00.getPosition(afwGeom.degrees) + afwGeom.Extent2D(0,side), afwGeom.degrees
        )
        p10 = self.wcs.skyToPixel(sky10) - p00
        p01 = self.wcs.skyToPixel(sky01) - p00

        area = side*side/abs(p10.getX()*p01.getY() - p01.getX()*p10.getY())

        self.assertAlmostEqual(math.sqrt(self.wcs.pixArea(p00)), math.sqrt(area))
        #
        # Now check that the area's the same as the CD matrix gives.
        #
        cd = [self.metadata.get("CD1_1"), self.metadata.get("CD1_2"),
              self.metadata.get("CD2_1"), self.metadata.get("CD2_2")]
        area = math.fabs(cd[0]*cd[3] - cd[1]*cd[2])

        self.assertAlmostEqual(math.sqrt(self.wcs.pixArea(p00)), math.sqrt(area))

    def testReadWcs(self):
        """Test reading a Wcs directly from a fits file"""

        meta = afwImage.readMetadata(InputImagePath + ".fits")
        wcs = afwImage.makeWcs(meta)

        sky0 = wcs.pixelToSky(0.0, 0.0).getPosition()
        sky1 = self.wcs.pixelToSky(0.0, 0.0).getPosition()
        self.assertEqual(sky0, sky1)

    def testShiftWcs(self):
        """Test shifting the reference pixel"""
        sky10_10 = self.wcs.pixelToSky(afwGeom.Point2D(10, 10))

        self.wcs.shiftReferencePixel(-10, -10)
        sky00 = self.wcs.pixelToSky(afwGeom.Point2D(0, 0))
        self.assertEqual((sky00[0], sky00[1]), (sky10_10[0], sky10_10[1]))

    def testCloneWcs(self):
        """Test Cloning a Wcs"""
        sky00 = self.wcs.pixelToSky(afwGeom.Point2D(0, 0)).getPosition()

        new = self.wcs.clone()
        self.wcs.pixelToSky(afwGeom.Point2D(10, 10)) # shouldn't affect new

        nsky00 = new.pixelToSky(afwGeom.Point2D(0, 0)).getPosition()
        self.assertEqual((sky00[0], sky00[1]), (nsky00[0], nsky00[1]))

    def testCD(self):
        cd = self.wcs.getCDMatrix()
        self.assertAlmostEqual(cd[0,0], self.metadata.getAsDouble("CD1_1"))
        self.assertAlmostEqual(cd[0,1], self.metadata.getAsDouble("CD1_2"))
        self.assertAlmostEqual(cd[1,0], self.metadata.getAsDouble("CD2_1"))
        self.assertAlmostEqual(cd[1,1], self.metadata.getAsDouble("CD2_2"))

    def testConstructor(self):
        afwImage.Wcs(self.wcs.getSkyOrigin().getPosition(afwGeom.degrees), self.wcs.getPixelOrigin(),
                            self.wcs.getCDMatrix())

    def testAffineTransform(self):
        sky00g = afwGeom.Point2D(10, 10)
        sky00c = afwCoord.makeCoord(afwCoord.ICRS, sky00g, afwGeom.degrees)
        a = self.wcs.linearizeSkyToPixel(sky00c)
        pix00g = self.wcs.skyToPixel(sky00c)
        pix00gApprox = a(sky00g);
        self.assertAlmostEqual(pix00g.getX(), pix00gApprox.getX())
        self.assertAlmostEqual(pix00g.getY(), pix00gApprox.getY())
        b = a.invert()
        self.assertAlmostEqual(self.wcs.pixArea(sky00g), abs(b[b.XX]* b[b.YY] - b[b.XY]*b[b.YX]))

class TestWcsCompare(unittest.TestCase):

    def setUp(self):
        crval = afwGeom.Point2D(1.23, 5.67)
        crpix = afwGeom.Point2D(102., 201.)
        cd = numpy.array([[5.399452e-5, -1.30770e-5], [1.30770e-5, 5.399452e-5]], dtype=float)
        self.plainWcs = afwImage.Wcs(crval, crpix, cd)
        self.sipWcs = afwImage.TanWcs(crval, crpix, cd)
        self.distortedWcs = afwImage.TanWcs(crval, crpix, cd, cd, cd, cd, cd)

    def tearDown(self):
        del self.plainWcs
        del self.sipWcs
        del self.distortedWcs

    def testEqualityCompare(self):
        self.assertNotEqual(self.plainWcs, self.sipWcs)
        self.assertNotEqual(self.sipWcs, self.plainWcs)
        self.assertNotEqual(self.distortedWcs, self.sipWcs)
        self.assertNotEqual(self.sipWcs, self.distortedWcs)
        plainWcsCopy = self.plainWcs.clone()
        sipWcsCopy = self.sipWcs.clone()
        distortedWcsCopy = self.distortedWcs.clone()
        self.assertEqual(plainWcsCopy, self.plainWcs)
        self.assertEqual(sipWcsCopy, self.sipWcs)
        self.assertEqual(distortedWcsCopy, self.distortedWcs)
        self.assertEqual(self.plainWcs, plainWcsCopy)
        self.assertEqual(self.sipWcs, sipWcsCopy)
        self.assertEqual(self.distortedWcs, distortedWcsCopy)
        self.assertNotEqual(plainWcsCopy, sipWcsCopy)
        self.assertNotEqual(sipWcsCopy, plainWcsCopy)
        self.assertNotEqual(distortedWcsCopy, sipWcsCopy)
        self.assertNotEqual(sipWcsCopy, distortedWcsCopy)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(WcsTestCase)
    suites += unittest.makeSuite(WCSTestCaseSDSS)
    suites += unittest.makeSuite(TestWcsCompare)
    suites += unittest.makeSuite(WCSTestCaseCFHT)
    suites += unittest.makeSuite(WCSRotateFlip)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
