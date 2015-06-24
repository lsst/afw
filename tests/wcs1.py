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
InputImagePath = os.path.join(dataDir, "871034p_1_MI")
InputSmallImagePath = os.path.join(dataDir, "data", "small_img.fits")
InputCorruptMaskedImageName = "small_MI_corrupt"
currDir = os.path.abspath(os.path.dirname(__file__))
InputCorruptFilePath = os.path.join(currDir, "data", InputCorruptMaskedImageName)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class WcsTestCase(unittest.TestCase):
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
        wcs = afwImage.makeWcs(crval, crpix, 5.399452e-5, -1.30770e-5, 1.30770e-5, 5.399452e-5)

    def testRotation(self):
        q1 = {0:afwGeom.Point2D(100., 1600.), 
              1:afwGeom.Point2D(self.size.getY() - 1600., 100.), 
              2:afwGeom.Point2D(self.size.getX() - 100., self.size.getY() - 1600.), 
              3:afwGeom.Point2D(1600., self.size.getX() - 100.)} 
        wcs = self.makeWcs()
        pos0 = self.wcs.pixelToSky(q1[0])
        for rot in (1,2,3):
            wcs = self.makeWcs()
            wcs.rotateImageBy90(rot, self.size)
            self.assertEqual(pos0, self.wcs.pixelToSky(q1[rot]))

    def testFlip(self):
        q1 = {'noFlip': afwGeom.Point2D(300., 900.),
              'flipLR': afwGeom.Point2D(self.size.getX()-300., 900.),
              'flipTB': afwGeom.Point2D(300., self.size.getY()-900.)}
        wcs = self.makeWcs()
        pos0 = self.wcs.pixelToSky(q1['noFlip'])
        wcs = self.makeWcs()
        wcs.flipImage(True, False, self.size)
        self.assertEqual(pos0, self.wcs.pixelToSky(q1['flipLR']))
        wcs = self.makeWcs()
        wcs.flipImage(False, True, self.size)
        self.assertEqual(pos0, self.wcs.pixelToSky(q1['flipTB']))

class WCSTestCaseSDSS(unittest.TestCase):
    """A test case for WCS using a small (SDSS) image with a slightly weird WCS"""

    def setUp(self):
        self.im = afwImage.DecoratedImageD(InputSmallImagePath)

        self.wcs = afwImage.makeWcs(self.im.getMetadata())

        if False:
            ds9.mtv(im, wcs=self.wcs)

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
        self.assertFalse(numpy.all(wcs.getCDMatrix() == numpy.array([[1.0, 0.0], [0.0, 1.0]])))

        md.set("PC1_1", 2)
        wcs = afwImage.makeWcs(md)
        self.assertFalse(numpy.all(wcs.getCDMatrix() == numpy.array([[2.0, 0.0], [0.0, 1.0]])))

    def testStripKeywords(self):
        """Test that we can strip WCS keywords from metadata when constructing a Wcs"""
        metadata = self.im.getMetadata()
        self.wcs = afwImage.makeWcs(metadata)

        self.assertFalse(metadata.exists("CRPIX1"))

        strip = True
        self.wcs = afwImage.makeWcs(metadata, strip)
        self.assertFalse(metadata.exists("CRPIX1"))

    def testAffineTransform(self):
        a = self.wcs.getLinearTransform()
        l = self.wcs.getCDMatrix()
        #print print a[a.XX], a[a.XY], a[a.YX], a[a.YY]
        #print a, l

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
        path = InputImagePath + "_img.fits"
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
        degrees = afwCoord.DEGREES
        sky10 = afwCoord.makeCoord(icrs, sky00 + afwGeom.Extent2D(side/cosdec, 0), degrees)
        sky01 = afwCoord.makeCoord(icrs, sky00 + afwGeom.Extent2D(0,side),         degrees)
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

        meta = afwImage.readMetadata(InputImagePath + "_img.fits")
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
        copy = afwImage.Wcs(self.wcs.getSkyOrigin(), self.wcs.getPixelOrigin(), 
                            self.wcs.getCDMatrix())

    def testAffineTransform(self):
        a = self.wcs.getLinearTransform()
        l = self.wcs.getCDMatrix()
        #print print a[a.XX], a[a.XY], a[a.YX], a[a.YY]

        sky00g = afwGeom.Point2D(10, 10)
        sky00i = afwGeom.Point2D(sky00g.getX(), sky00g.getY())
        sky00c = afwCoord.makeCoord(afwCoord.ICRS, sky00i, afwCoord.DEGREES)
        a = self.wcs.linearizeSkyToPixel(sky00c)
        pix00i = self.wcs.skyToPixel(sky00c)
        pix00g = afwGeom.Point2D(pix00i.getX(), pix00i.getY())
        sky00gApprox = a(pix00g);
        self.assertAlmostEqual(sky00g.getX(), sky00gApprox.getX())
        self.assertAlmostEqual(sky00g.getY(), sky00gApprox.getY())
        self.assertAlmostEqual(self.wcs.pixArea(sky00i), abs(a[a.XX]* a[a.YY] - a[a.XY]*a[a.YX]))
        a.invert()

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
#    suites += unittest.makeSuite(WCSTestCaseCFHT)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
