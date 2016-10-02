#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see <http://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#
#pybind11#import os.path
#pybind11#import math
#pybind11#import unittest
#pybind11#import numpy
#pybind11#
#pybind11#import lsst.utils
#pybind11#import lsst.utils.tests
#pybind11#import lsst.daf.base as dafBase
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.coord as afwCoord
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#import lsst
#pybind11#
#pybind11#try:
#pybind11#    type(verbose)
#pybind11#except NameError:
#pybind11#    verbose = 0
#pybind11#
#pybind11#try:
#pybind11#    afwdataDir = lsst.utils.getPackageDir("afwdata")
#pybind11#except pexExcept.NotFoundError:
#pybind11#    afwdataDir = None
#pybind11#    InputImagePath = None
#pybind11#    InputSmallImagePath = None
#pybind11#else:
#pybind11#    InputImagePath = os.path.join(afwdataDir, "data", "871034p_1_MI")
#pybind11#    InputSmallImagePath = os.path.join(afwdataDir, "data", "small_img.fits")
#pybind11#
#pybind11#InputCorruptMaskedImageName = "small_MI_corrupt"
#pybind11#currDir = os.path.abspath(os.path.dirname(__file__))
#pybind11#InputCorruptFilePath = os.path.join(currDir, "data", InputCorruptMaskedImageName)
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#CoordSysList = [afwCoord.ICRS, afwCoord.FK5, afwCoord.ECLIPTIC, afwCoord.GALACTIC]
#pybind11#
#pybind11#
#pybind11#def coordSysEquinoxIter(equinoxList=(1950, 1977, 2000)):
#pybind11#    for coordSys in CoordSysList:
#pybind11#        if coordSys in (afwCoord.FK5, afwCoord.ECLIPTIC):
#pybind11#            for equinox in equinoxList:
#pybind11#                yield (coordSys, equinox)
#pybind11#        else:
#pybind11#            yield (coordSys, 2000)  # Coord's default
#pybind11#
#pybind11#
#pybind11#def makeWcs(
#pybind11#    pixelScale=0.2*afwGeom.arcseconds,
#pybind11#    crPixPos=(2000.0, 1520.5),
#pybind11#    crValDeg=(27.53, 87.123),
#pybind11#    posAng=afwGeom.Angle(0.0),
#pybind11#    doFlipX=False,
#pybind11#    projection="TAN",
#pybind11#    coordSys=afwCoord.ICRS,
#pybind11#    equinox=2000
#pybind11#):
#pybind11#    """Make an simple TAN WCS with sensible defaults
#pybind11#
#pybind11#    @param[in] pixelScale: desired scale, as sky/pixel, an afwGeom.Angle
#pybind11#    @param[in] crPixPos: crPix for WCS, using the LSST standard; a pair of floats
#pybind11#    @param[in] crValDeg: crVal for WCS, in degrees; a pair of floats
#pybind11#    @param[in] posAng: position angle (afwGeom.Angle)
#pybind11#    @param[in] doFlipX: flip X axis?
#pybind11#    @param[in] projection: WCS projection (e.g. "TAN" or "STG")
#pybind11#    @param[in] coordSys: coordinate system enum
#pybind11#    @param[in] equinox: date of equinox; should be 2000 for ICRS or GALACTIC
#pybind11#    """
#pybind11#    if len(projection) != 3:
#pybind11#        raise RuntimeError("projection=%r; must have length 3" % (projection,))
#pybind11#
#pybind11#    csysPrefixes, radesys = {
#pybind11#        afwCoord.ICRS: (("RA", "DEC"), "ICRS"),
#pybind11#        afwCoord.FK5: (("RA", "DEC"), "FK5"),
#pybind11#        afwCoord.ECLIPTIC: (("ELON", "ELAT"), None),
#pybind11#        afwCoord.GALACTIC: (("GLON", "GLAT"), None),
#pybind11#    }[coordSys]
#pybind11#
#pybind11#    ctypeList = [("%-4s-%3s" % (csysPrefixes[i], projection)).replace(" ", "-") for i in range(2)]
#pybind11#    ps = dafBase.PropertySet()
#pybind11#    crPixFits = [ind + 1.0 for ind in crPixPos]  # convert pix position to FITS standard
#pybind11#    posAngRad = posAng.asRadians()
#pybind11#    pixelScaleDeg = pixelScale.asDegrees()
#pybind11#    cdMat = numpy.array([[math.cos(posAngRad), math.sin(posAngRad)],
#pybind11#                         [-math.sin(posAngRad), math.cos(posAngRad)]], dtype=float) * pixelScaleDeg
#pybind11#    if doFlipX:
#pybind11#        cdMat[:, 0] = -cdMat[:, 0]
#pybind11#    for i in range(2):
#pybind11#        ip1 = i + 1
#pybind11#        ps.add("CTYPE%1d" % (ip1,), ctypeList[i])
#pybind11#        ps.add("CRPIX%1d" % (ip1,), crPixFits[i])
#pybind11#        ps.add("CRVAL%1d" % (ip1,), crValDeg[i])
#pybind11#    if radesys:
#pybind11#        ps.add("RADESYS", radesys)
#pybind11#    ps.add("EQUINOX", equinox)
#pybind11#    ps.add("CD1_1", cdMat[0, 0])
#pybind11#    ps.add("CD2_1", cdMat[1, 0])
#pybind11#    ps.add("CD1_2", cdMat[0, 1])
#pybind11#    ps.add("CD2_2", cdMat[1, 1])
#pybind11#    return afwImage.makeWcs(ps)
#pybind11#
#pybind11#
#pybind11#def localMakeCoord(coordSys, posDeg, equinox):
#pybind11#    """Make a coord, ignoring equinox if necessary
#pybind11#    """
#pybind11#    if coordSys in (afwCoord.ICRS, afwCoord.GALACTIC):
#pybind11#        return afwCoord.makeCoord(coordSys, posDeg[0]*afwGeom.degrees, posDeg[1]*afwGeom.degrees)
#pybind11#    return afwCoord.makeCoord(coordSys, posDeg[0]*afwGeom.degrees, posDeg[1]*afwGeom.degrees, equinox)
#pybind11#
#pybind11#
#pybind11#class WcsTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def testCD_PC(self):
#pybind11#        """Test that we can read a FITS file with both CD and PC keys (like early Suprimecam files)"""
#pybind11#        md = dafBase.PropertyList()
#pybind11#        for k, v in (
#pybind11#            ("EQUINOX", 2000.0),
#pybind11#            ("RADESYS", 'FK5'),
#pybind11#            ("CRPIX1", 5353.0),
#pybind11#            ("CRPIX2", -35.0),
#pybind11#            ("CD1_1", 0.0),
#pybind11#            ("CD1_2", -5.611E-05),
#pybind11#            ("CD2_1", -5.611E-05),
#pybind11#            ("CD2_2", -0.0),
#pybind11#            ("CRVAL1", 4.5789875),
#pybind11#            ("CRVAL2", 16.30004444),
#pybind11#            ("CUNIT1", 'deg'),
#pybind11#            ("CUNIT2", 'deg'),
#pybind11#            ("CTYPE1", 'RA---TAN'),
#pybind11#            ("CTYPE2", 'DEC--TAN'),
#pybind11#            ("CDELT1", -5.611E-05),
#pybind11#            ("CDELT2", 5.611E-05),
#pybind11#        ):
#pybind11#            md.set(k, v)
#pybind11#
#pybind11#        wcs = afwImage.makeWcs(md)
#pybind11#
#pybind11#        x, y = 1000, 2000
#pybind11#        ra, dec = 4.459815023498577, 16.544199850984768
#pybind11#
#pybind11#        sky = wcs.pixelToSky(x, y)
#pybind11#        for i, v in enumerate([ra, dec]):
#pybind11#            self.assertEqual(sky[i].asDegrees(), v)
#pybind11#
#pybind11#        for badPC in (False, True):
#pybind11#            if verbose:
#pybind11#                print("Checking PC coefficients: badPC =", badPC)
#pybind11#            for k, v in (
#pybind11#                ("PC001001", 0.0),
#pybind11#                ("PC001002", -1.0 if badPC else 1.0),
#pybind11#                ("PC002001", 1.0 if badPC else -1.0),
#pybind11#                ("PC002002", 0.0),
#pybind11#            ):
#pybind11#                md.set(k, v)
#pybind11#
#pybind11#            # Check Greisen and Calabretta A&A 395 1061 (2002), Eq. 3
#pybind11#            if not badPC:
#pybind11#                for i in (1, 2,):
#pybind11#                    for j in (1, 2,):
#pybind11#                        self.assertEqual(md.get("CD%d_%d" % (i, j)),
#pybind11#                                         md.get("CDELT%d" % i)*md.get("PC00%d00%d" % (i, j)))
#pybind11#
#pybind11#            wcs = afwImage.makeWcs(md)
#pybind11#            sky = wcs.pixelToSky(x, y)
#pybind11#            for i, v in enumerate([ra, dec]):
#pybind11#                self.assertEqual(sky[i].asDegrees(), v)
#pybind11#
#pybind11#    def testCast(self):
#pybind11#        # strangely, this overload of makeWcs returns a TAN WCS that's not a TanWcs
#pybind11#        wcs = afwImage.makeWcs(afwCoord.IcrsCoord(45.0*afwGeom.degrees, 45.0*afwGeom.degrees),
#pybind11#                               afwGeom.Point2D(0.0, 0.0), 1.0, 0.0, 0.0, 1.0)
#pybind11#        # ...but if you round-trip it through a PropertySet, you get a TanWcs in a Wcs ptr,
#pybind11#        # which is what we want for this test.
#pybind11#        base = afwImage.makeWcs(wcs.getFitsMetadata())
#pybind11#        self.assertEqual(type(base), afwImage.Wcs)
#pybind11#        derived = afwImage.TanWcs.cast(base)
#pybind11#        self.assertEqual(type(derived), afwImage.TanWcs)
#pybind11#
#pybind11#    def testCoordConversion(self):
#pybind11#        """Test that skyToPixel and pixelToSky handle coordinate system and equinox correctly
#pybind11#
#pybind11#        Given two WCS that are identical except perhaps for coordinate system and equinox
#pybind11#        compute: sky2 = wcs2.pixelToSky(wcs1.skyToPixel(sky1)
#pybind11#        The result should be the same numerical values, but with wcs2's coordinate system and equinox
#pybind11#        """
#pybind11#        crValDeg = (27.53, 87.123)
#pybind11#        for coordSys1, equinox1 in coordSysEquinoxIter():
#pybind11#            wcs1 = makeWcs(crValDeg=crValDeg, coordSys=coordSys1, equinox=equinox1)
#pybind11#            for coordSys2, equinox2 in coordSysEquinoxIter():
#pybind11#                wcs2 = makeWcs(crValDeg=crValDeg, coordSys=coordSys2, equinox=equinox2)
#pybind11#                for sky1Deg in (crValDeg, (18.52, 46.765)):
#pybind11#                    coord1 = localMakeCoord(coordSys1, sky1Deg, equinox1)
#pybind11#                    pixPos = wcs1.skyToPixel(coord1)
#pybind11#                    coord2 = wcs2.pixelToSky(pixPos)
#pybind11#
#pybind11#                    desCoord2 = localMakeCoord(
#pybind11#                        coordSys2, (coord1[0].asDegrees(), coord1[1].asDegrees()), equinox2)
#pybind11#                    self.assertCoordsNearlyEqual(coord2, desCoord2)
#pybind11#
#pybind11#    def testGetCoordSys(self):
#pybind11#        """Test getCoordSystem, getEquinox"""
#pybind11#        def isIcrs(wcs):
#pybind11#            """Return True if wcs is ICRS or FK5 J2000"""
#pybind11#            csys = wcs.getCoordSystem()
#pybind11#            if csys == afwCoord.ICRS:
#pybind11#                return True
#pybind11#            return csys == afwCoord.FK5 and wcs.getEquinox() == 2000
#pybind11#
#pybind11#        def refIsSameSkySystem(wcs1, wcs2):
#pybind11#            if isIcrs(wcs1) and isIcrs(wcs2):
#pybind11#                return True
#pybind11#            return (wcs1.getCoordSystem() == wcs2.getCoordSystem()) and (wcs1.getEquinox() == wcs2.getEquinox())
#pybind11#
#pybind11#        for coordSys, equinox in coordSysEquinoxIter():
#pybind11#            wcs = makeWcs(coordSys=coordSys, equinox=equinox)
#pybind11#            self.assertEqual(wcs.getCoordSystem(), coordSys)
#pybind11#            if coordSys not in (afwCoord.ICRS, afwCoord.GALACTIC):
#pybind11#                # ICRS and Galactic by definition does not have an equinox.
#pybind11#                self.assertEqual(wcs.getEquinox(), equinox)
#pybind11#            predIsIcrs = coordSys == afwCoord.ICRS or (coordSys == afwCoord.FK5 and equinox == 2000)
#pybind11#            self.assertEqual(predIsIcrs, isIcrs(wcs))
#pybind11#            for coordSys2, equinox2 in coordSysEquinoxIter():
#pybind11#                wcs2 = makeWcs(coordSys=coordSys2, equinox=equinox2)
#pybind11#                self.assertEqual(refIsSameSkySystem(wcs, wcs2), wcs.isSameSkySystem(wcs2))
#pybind11#
#pybind11#    def testIcrsEquinox(self):
#pybind11#        """Check that EQUINOX is not written to FITS for ICRS coordinates"""
#pybind11#        def checkEquinoxHeader(coordSysName, writeEquinox):
#pybind11#            coordSys = getattr(afwCoord, coordSysName)
#pybind11#            # We should get the same behaviour with both Wcs and TanWcs: check them both.
#pybind11#            for dummyWcs in (makeWcs(coordSys=coordSys, projection="SIN"),
#pybind11#                             makeWcs(coordSys=coordSys, projection="TAN")):
#pybind11#                dummyExposure = afwImage.ExposureF()
#pybind11#                dummyExposure.setWcs(dummyWcs)
#pybind11#                with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#                    dummyExposure.writeFits(tmpFile)
#pybind11#                    metadata = afwImage.readMetadata(tmpFile)
#pybind11#                    self.assertEqual(metadata.get("RADESYS"), coordSysName)
#pybind11#                    self.assertTrue(("EQUINOX" in metadata.names()) == writeEquinox)
#pybind11#        checkEquinoxHeader("ICRS", False)
#pybind11#        checkEquinoxHeader("FK5", True)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class WCSRotateFlip(unittest.TestCase):
#pybind11#    """A test case for the methods to rotate and flip a wcs under similar operations to the image pixels"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.size = afwGeom.Extent2I(509, 2000)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.size
#pybind11#
#pybind11#    def makeWcs(self):
#pybind11#        crval = afwCoord.Coord(afwGeom.Point2D(1.606631, 5.090329))
#pybind11#        crpix = afwGeom.Point2D(2036., 2000.)
#pybind11#        return afwImage.makeWcs(crval, crpix, 5.399452e-5, -1.30770e-5, 1.30770e-5, 5.399452e-5)
#pybind11#
#pybind11#    def testRotation(self):
#pybind11#        # Origin for LSST pixels is (0,0).  Need to subtract one when rotating to avoid off by one.
#pybind11#        # E.g. UR (507, 1999) goes to (0,0) for nRot = 2
#pybind11#        q1 = {0: afwGeom.Point2D(100., 1600.),
#pybind11#              1: afwGeom.Point2D(self.size.getY() - 1600. - 1, 100.),
#pybind11#              2: afwGeom.Point2D(self.size.getX() - 100. - 1, self.size.getY() - 1600. - 1),
#pybind11#              3: afwGeom.Point2D(1600., self.size.getX() - 100. - 1)}
#pybind11#        wcs = self.makeWcs()
#pybind11#        pos0 = wcs.pixelToSky(q1[0])
#pybind11#        for rot in (1, 2, 3):
#pybind11#            wcs = self.makeWcs()
#pybind11#            wcs.rotateImageBy90(rot, self.size)
#pybind11#            self.assertEqual(pos0, wcs.pixelToSky(q1[rot]))
#pybind11#
#pybind11#    def testFlip(self):
#pybind11#        q1 = {'noFlip': afwGeom.Point2D(300., 900.),
#pybind11#              'flipLR': afwGeom.Point2D(self.size.getX()-300.-1, 900.),
#pybind11#              'flipTB': afwGeom.Point2D(300., self.size.getY()-900.-1)}
#pybind11#        wcs = self.makeWcs()
#pybind11#        pos0 = wcs.pixelToSky(q1['noFlip'])
#pybind11#        wcs = self.makeWcs()
#pybind11#        wcs.flipImage(True, False, self.size)
#pybind11#        self.assertEqual(pos0, wcs.pixelToSky(q1['flipLR']))
#pybind11#        wcs = self.makeWcs()
#pybind11#        wcs.flipImage(False, True, self.size)
#pybind11#        self.assertEqual(pos0, wcs.pixelToSky(q1['flipTB']))
#pybind11#
#pybind11#
#pybind11#class WCSTestCaseSDSS(unittest.TestCase):
#pybind11#    """A test case for WCS using a small (SDSS) image with a slightly weird WCS"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        if InputSmallImagePath is not None:
#pybind11#            self.im = afwImage.DecoratedImageD(InputSmallImagePath)
#pybind11#
#pybind11#            self.wcs = afwImage.makeWcs(self.im.getMetadata())
#pybind11#        else:
#pybind11#            self.im = None
#pybind11#            self.wcs = None
#pybind11#
#pybind11#            if False:
#pybind11#                ds9.mtv(self.im, wcs=self.wcs)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        if self.im is not None:
#pybind11#            del self.im
#pybind11#        if self.wcs is not None:
#pybind11#            del self.wcs
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testCrpix(self):
#pybind11#        metadata = self.im.getMetadata()
#pybind11#        crpix0 = metadata.getAsDouble("CRPIX1")
#pybind11#        crpix1 = metadata.getAsDouble("CRPIX2")
#pybind11#
#pybind11#        lsstCrpix = self.wcs.getPixelOrigin()
#pybind11#
#pybind11#        self.assertEqual(lsstCrpix[0], crpix0-1)
#pybind11#        self.assertEqual(lsstCrpix[1], crpix1-1)
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testXyToRaDecArguments(self):
#pybind11#        """Check that conversion of xy to ra dec (and back again) works"""
#pybind11#        xy = afwGeom.Point2D(110, 123)
#pybind11#        raDec = self.wcs.pixelToSky(xy)
#pybind11#        xy2 = self.wcs.skyToPixel(raDec)
#pybind11#
#pybind11#        self.assertAlmostEqual(xy.getX(), xy2.getX())
#pybind11#        self.assertAlmostEqual(xy.getY(), xy2.getY())
#pybind11#
#pybind11#        raDec = afwCoord.makeCoord(afwCoord.ICRS, 245.167400 * afwGeom.degrees, +19.1976583 * afwGeom.degrees)
#pybind11#
#pybind11#        xy = self.wcs.skyToPixel(raDec)
#pybind11#        raDec2 = self.wcs.pixelToSky(xy)
#pybind11#
#pybind11#        self.assertAlmostEqual(raDec[0].asDegrees(), raDec2[0].asDegrees())
#pybind11#        self.assertAlmostEqual(raDec[1].asDegrees(), raDec2[1].asDegrees())
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def test_RaTan_DecTan(self):
#pybind11#        """Check the RA---TAN, DEC--TAN WCS conversion"""
#pybind11#        # values from wcstools xy2sky (v3.8.1). Confirmed by ds9
#pybind11#        raDec0 = afwGeom.Point2D(245.15984167, +19.1960472)
#pybind11#        raDec = self.wcs.pixelToSky(0.0, 0.0).getPosition()
#pybind11#
#pybind11#        self.assertAlmostEqual(raDec.getX(), raDec0.getX(), 5)
#pybind11#        self.assertAlmostEqual(raDec.getY(), raDec0.getY(), 5)
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testIdentity(self):
#pybind11#        """Convert from ra, dec to col, row and back again"""
#pybind11#        raDec = afwCoord.makeCoord(afwCoord.ICRS, 244 * afwGeom.degrees, 20 * afwGeom.degrees)
#pybind11#        if verbose:
#pybind11#            print('testIdentity')
#pybind11#            print('wcs:')
#pybind11#            for x in self.wcs.getFitsMetadata().toList():
#pybind11#                print('  ', x)
#pybind11#            print('raDec:', raDec)
#pybind11#            print(type(self.wcs))
#pybind11#        rowCol = self.wcs.skyToPixel(raDec)
#pybind11#        raDec2 = self.wcs.pixelToSky(rowCol)
#pybind11#
#pybind11#        if verbose:
#pybind11#            print('rowCol:', rowCol)
#pybind11#            print('raDec2:', raDec2)
#pybind11#
#pybind11#        p1 = raDec.getPosition()
#pybind11#        p2 = raDec.getPosition()
#pybind11#        if verbose:
#pybind11#            print('p1,p2', p1, p2)
#pybind11#        self.assertAlmostEqual(p1[0], p2[0])
#pybind11#        self.assertAlmostEqual(p1[1], p2[1])
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testInvalidRaDec(self):
#pybind11#        """Test a conversion for an invalid position.  Well, "test" isn't
#pybind11#        quite right as the result is invalid, but make sure that it still is"""
#pybind11#        raDec = afwCoord.makeCoord(afwCoord.ICRS, 1 * afwGeom.degrees, 2 * afwGeom.degrees)
#pybind11#
#pybind11#        with self.assertRaises(lsst.pex.exceptions.Exception):
#pybind11#            self.wcs.skyToPixel(raDec)
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testCD(self):
#pybind11#        self.wcs.getCDMatrix()
#pybind11#
#pybind11#    def testCreateCDMatrix(self):
#pybind11#        """Test that we make a correct CD matrix even if the header only has a PC matrix"""
#pybind11#        md = dafBase.PropertySet()
#pybind11#        md.set("NAXIS", 2)
#pybind11#        md.set("CTYPE1", "RA---TAN")
#pybind11#        md.set("CTYPE2", "DEC--TAN")
#pybind11#        md.set("CRPIX1", 0)
#pybind11#        md.set("CRPIX2", 0)
#pybind11#        md.set("CRVAL1", 0)
#pybind11#        md.set("CRVAL2", 0)
#pybind11#        md.set("RADECSYS", "FK5")
#pybind11#        md.set("EQUINOX", 2000.0)
#pybind11#
#pybind11#        wcs = afwImage.makeWcs(md)
#pybind11#        self.assertTrue(numpy.all(wcs.getCDMatrix() == numpy.array([[1.0, 0.0], [0.0, 1.0]])))
#pybind11#
#pybind11#        md.set("PC1_1", 2)
#pybind11#        wcs = afwImage.makeWcs(md)
#pybind11#        self.assertTrue(numpy.all(wcs.getCDMatrix() == numpy.array([[2.0, 0.0], [0.0, 1.0]])))
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testStripKeywords(self):
#pybind11#        """Test that we can strip WCS keywords from metadata when constructing a Wcs"""
#pybind11#        metadata = self.im.getMetadata()
#pybind11#        self.wcs = afwImage.makeWcs(metadata)
#pybind11#
#pybind11#        self.assertTrue(metadata.exists("CRPIX1"))
#pybind11#
#pybind11#        strip = True
#pybind11#        self.wcs = afwImage.makeWcs(metadata, strip)
#pybind11#        self.assertFalse(metadata.exists("CRPIX1"))
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testAffineTransform(self):
#pybind11#        self.wcs.getLinearTransform()
#pybind11#        self.wcs.getCDMatrix()
#pybind11#
#pybind11#    def testXY0(self):
#pybind11#        """Test that XY0 values are handled correctly when building an exposure and also when
#pybind11#        reading the WCS header directly.  #2205"""
#pybind11#        bbox = afwGeom.Box2I(afwGeom.Point2I(1000, 1000), afwGeom.Extent2I(10, 10))
#pybind11#
#pybind11#        def makeWcs(crPixPos, crValDeg, projection):
#pybind11#            ps = dafBase.PropertySet()
#pybind11#            ctypes = [("%-5s%3s" % (("RA", "DEC")[i], projection)).replace(" ", "-") for i in range(2)]
#pybind11#            for i in range(2):
#pybind11#                ip1 = i + 1
#pybind11#                ps.add("CTYPE%1d" % (ip1,), ctypes[i])
#pybind11#                ps.add("CRPIX%1d" % (ip1,), crPixPos[i])
#pybind11#                ps.add("CRVAL%1d" % (ip1,), crValDeg[i])
#pybind11#            ps.add("RADECSYS", "ICRS")
#pybind11#            ps.add("EQUINOX", 2000)
#pybind11#            ps.add("CD1_1", -0.001)
#pybind11#            ps.add("CD2_1", 0.0)
#pybind11#            ps.add("CD1_2", 0.0)
#pybind11#            ps.add("CD2_2", 0.001)
#pybind11#            return afwImage.makeWcs(ps)
#pybind11#
#pybind11#        wcs = makeWcs(
#pybind11#            crPixPos=(100.0, 100.0),
#pybind11#            crValDeg=(10.0, 35.0),
#pybind11#            projection="STG",  # also fails for TAN
#pybind11#        )
#pybind11#
#pybind11#        exposure = afwImage.ExposureF(bbox, wcs)
#pybind11#        pixPos = afwGeom.Box2D(bbox).getMax()
#pybind11#        if verbose:
#pybind11#            print("XY0=", exposure.getXY0())
#pybind11#            print("pixPos=", pixPos)
#pybind11#        skyPos = wcs.pixelToSky(pixPos)
#pybind11#
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            exposure.writeFits(tmpFile)
#pybind11#            for useExposure in (False, True):
#pybind11#                if useExposure:
#pybind11#                    unpExp = afwImage.ExposureF(tmpFile)
#pybind11#                    unpWcs = unpExp.getWcs()
#pybind11#                else:
#pybind11#                    md = afwImage.readMetadata(tmpFile)
#pybind11#                    unpWcs = afwImage.makeWcs(md, False)
#pybind11#                unpPixPos = unpWcs.skyToPixel(skyPos)
#pybind11#
#pybind11#                if verbose:
#pybind11#                    print("useExposure=%s; unp pixPos=%s" % (useExposure, unpPixPos))
#pybind11#
#pybind11#                for i in range(2):
#pybind11#                    self.assertAlmostEqual(unpPixPos[i], 1009.5)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#@unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#class WCSTestCaseCFHT(unittest.TestCase):
#pybind11#    """A test case for WCS"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        if InputImagePath is not None:
#pybind11#            path = InputImagePath + ".fits"
#pybind11#            self.metadata = afwImage.readMetadata(path)
#pybind11#            self.wcs = afwImage.makeWcs(self.metadata)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        if InputImagePath is not None:
#pybind11#            del self.wcs
#pybind11#            del self.metadata
#pybind11#
#pybind11#    def test_RaTan_DecTan(self):
#pybind11#        """Check the RA---TAN, DEC--TAN WCS conversion"""
#pybind11#        raDec = self.wcs.pixelToSky(0.0, 0.0).getPosition()  # position read off ds9
#pybind11#
#pybind11#        self.assertAlmostEqual(raDec[0], 17.87673, 5)  # ra from ds9
#pybind11#        self.assertAlmostEqual(raDec[1], 7.72231, 5)  # dec from ds9
#pybind11#
#pybind11#    def testPlateScale(self):
#pybind11#        """Test that we can measure the area of a pixel"""
#pybind11#
#pybind11#        p00 = afwGeom.Point2D(10, 10)
#pybind11#        p00 = afwGeom.Point2D(self.metadata.getAsDouble("CRPIX1"), self.metadata.getAsDouble("CRPIX2"))
#pybind11#
#pybind11#        sky00 = self.wcs.pixelToSky(p00)
#pybind11#        cosdec = math.cos(sky00[1])
#pybind11#
#pybind11#        side = 1e-3
#pybind11#        icrs = afwCoord.ICRS
#pybind11#        sky10 = afwCoord.makeCoord(
#pybind11#            icrs, sky00.getPosition(afwGeom.degrees) + afwGeom.Extent2D(side/cosdec, 0), afwGeom.degrees
#pybind11#        )
#pybind11#        sky01 = afwCoord.makeCoord(
#pybind11#            icrs, sky00.getPosition(afwGeom.degrees) + afwGeom.Extent2D(0, side), afwGeom.degrees
#pybind11#        )
#pybind11#        p10 = self.wcs.skyToPixel(sky10) - p00
#pybind11#        p01 = self.wcs.skyToPixel(sky01) - p00
#pybind11#
#pybind11#        area = side*side/abs(p10.getX()*p01.getY() - p01.getX()*p10.getY())
#pybind11#
#pybind11#        self.assertAlmostEqual(math.sqrt(self.wcs.pixArea(p00)), math.sqrt(area))
#pybind11#        #
#pybind11#        # Now check that the area's the same as the CD matrix gives.
#pybind11#        #
#pybind11#        cd = [self.metadata.get("CD1_1"), self.metadata.get("CD1_2"),
#pybind11#              self.metadata.get("CD2_1"), self.metadata.get("CD2_2")]
#pybind11#        area = math.fabs(cd[0]*cd[3] - cd[1]*cd[2])
#pybind11#
#pybind11#        self.assertAlmostEqual(math.sqrt(self.wcs.pixArea(p00)), math.sqrt(area))
#pybind11#
#pybind11#    def testReadWcs(self):
#pybind11#        """Test reading a Wcs directly from a fits file"""
#pybind11#
#pybind11#        meta = afwImage.readMetadata(InputImagePath + ".fits")
#pybind11#        wcs = afwImage.makeWcs(meta)
#pybind11#
#pybind11#        sky0 = wcs.pixelToSky(0.0, 0.0).getPosition()
#pybind11#        sky1 = self.wcs.pixelToSky(0.0, 0.0).getPosition()
#pybind11#        self.assertEqual(sky0, sky1)
#pybind11#
#pybind11#    def testShiftWcs(self):
#pybind11#        """Test shifting the reference pixel"""
#pybind11#        sky10_10 = self.wcs.pixelToSky(afwGeom.Point2D(10, 10))
#pybind11#
#pybind11#        self.wcs.shiftReferencePixel(-10, -10)
#pybind11#        sky00 = self.wcs.pixelToSky(afwGeom.Point2D(0, 0))
#pybind11#        self.assertEqual((sky00[0], sky00[1]), (sky10_10[0], sky10_10[1]))
#pybind11#
#pybind11#    def testCloneWcs(self):
#pybind11#        """Test Cloning a Wcs"""
#pybind11#        sky00 = self.wcs.pixelToSky(afwGeom.Point2D(0, 0)).getPosition()
#pybind11#
#pybind11#        new = self.wcs.clone()
#pybind11#        self.wcs.pixelToSky(afwGeom.Point2D(10, 10))  # shouldn't affect new
#pybind11#
#pybind11#        nsky00 = new.pixelToSky(afwGeom.Point2D(0, 0)).getPosition()
#pybind11#        self.assertEqual((sky00[0], sky00[1]), (nsky00[0], nsky00[1]))
#pybind11#
#pybind11#    def testCD(self):
#pybind11#        cd = self.wcs.getCDMatrix()
#pybind11#        self.assertAlmostEqual(cd[0, 0], self.metadata.getAsDouble("CD1_1"))
#pybind11#        self.assertAlmostEqual(cd[0, 1], self.metadata.getAsDouble("CD1_2"))
#pybind11#        self.assertAlmostEqual(cd[1, 0], self.metadata.getAsDouble("CD2_1"))
#pybind11#        self.assertAlmostEqual(cd[1, 1], self.metadata.getAsDouble("CD2_2"))
#pybind11#
#pybind11#    def testConstructor(self):
#pybind11#        afwImage.Wcs(self.wcs.getSkyOrigin().getPosition(afwGeom.degrees), self.wcs.getPixelOrigin(),
#pybind11#                     self.wcs.getCDMatrix())
#pybind11#
#pybind11#    def testAffineTransform(self):
#pybind11#        sky00g = afwGeom.Point2D(10, 10)
#pybind11#        sky00c = afwCoord.makeCoord(afwCoord.ICRS, sky00g, afwGeom.degrees)
#pybind11#        a = self.wcs.linearizeSkyToPixel(sky00c)
#pybind11#        pix00g = self.wcs.skyToPixel(sky00c)
#pybind11#        pix00gApprox = a(sky00g)
#pybind11#        self.assertAlmostEqual(pix00g.getX(), pix00gApprox.getX())
#pybind11#        self.assertAlmostEqual(pix00g.getY(), pix00gApprox.getY())
#pybind11#        b = a.invert()
#pybind11#        self.assertAlmostEqual(self.wcs.pixArea(sky00g), abs(b[b.XX] * b[b.YY] - b[b.XY]*b[b.YX]))
#pybind11#
#pybind11#
#pybind11#class TestWcsCompare(unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        crval = afwGeom.Point2D(1.23, 5.67)
#pybind11#        crpix = afwGeom.Point2D(102., 201.)
#pybind11#        cd = numpy.array([[5.399452e-5, -1.30770e-5], [1.30770e-5, 5.399452e-5]], dtype=float)
#pybind11#        self.plainWcs = afwImage.Wcs(crval, crpix, cd)
#pybind11#        self.sipWcs = afwImage.TanWcs(crval, crpix, cd)
#pybind11#        self.distortedWcs = afwImage.TanWcs(crval, crpix, cd, cd, cd, cd, cd)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.plainWcs
#pybind11#        del self.sipWcs
#pybind11#        del self.distortedWcs
#pybind11#
#pybind11#    def testEqualityCompare(self):
#pybind11#        self.assertNotEqual(self.plainWcs, self.sipWcs)
#pybind11#        self.assertNotEqual(self.sipWcs, self.plainWcs)
#pybind11#        self.assertNotEqual(self.distortedWcs, self.sipWcs)
#pybind11#        self.assertNotEqual(self.sipWcs, self.distortedWcs)
#pybind11#        plainWcsCopy = self.plainWcs.clone()
#pybind11#        sipWcsCopy = self.sipWcs.clone()
#pybind11#        distortedWcsCopy = self.distortedWcs.clone()
#pybind11#        self.assertEqual(plainWcsCopy, self.plainWcs)
#pybind11#        self.assertEqual(sipWcsCopy, self.sipWcs)
#pybind11#        self.assertEqual(distortedWcsCopy, self.distortedWcs)
#pybind11#        self.assertEqual(self.plainWcs, plainWcsCopy)
#pybind11#        self.assertEqual(self.sipWcs, sipWcsCopy)
#pybind11#        self.assertEqual(self.distortedWcs, distortedWcsCopy)
#pybind11#        self.assertNotEqual(plainWcsCopy, sipWcsCopy)
#pybind11#        self.assertNotEqual(sipWcsCopy, plainWcsCopy)
#pybind11#        self.assertNotEqual(distortedWcsCopy, sipWcsCopy)
#pybind11#        self.assertNotEqual(sipWcsCopy, distortedWcsCopy)
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
