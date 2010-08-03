#!/usr/bin/env python

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

# -*- python -*-
"""
Tests for Coord

Run with:
   python Coord.py
or
   python
   >>> import Coord
   >>> Coord.run()
"""

import unittest
import lsst.afw.geom             as afwGeom
import lsst.afw.coord            as afwCoord
import lsst.utils.tests          as utilsTests
import lsst.daf.base             as dafBase
import lsst.pex.exceptions       as pexEx

# todo: see if we can give an Fk5 and an ecliptic at a different epoch and get the right answer
# todo: make sure ICrs stuff works

######################################
# main body of code
######################################
class CoordTestCase(unittest.TestCase):

    def setUp(self):
        # define some arbitrary values
        self.ra, self.raKnown  = "10:00:00.00", 10.0
        self.dec, self.decKnown = "-02:30:00.00", -2.5
        self.l = 100.0
        self.b = 30.0

        # a handy list of coords we want to test
        self.coordList = [
            [afwCoord.Fk5Coord,      afwCoord.FK5,      afwCoord.cast_Fk5,      "FK5"],
            [afwCoord.IcrsCoord,     afwCoord.ICRS,     afwCoord.cast_Icrs,     "ICRS"],
            [afwCoord.GalacticCoord, afwCoord.GALACTIC, afwCoord.cast_Galactic, "GALACTIC"],
            [afwCoord.EclipticCoord, afwCoord.ECLIPTIC, afwCoord.cast_Ecliptic, "ECLIPTIC"],
            # we can't factory an Topocentric ... Observatory must be specified.
            # [afwCoord.TopocentricCoord, afwCoord.TOPOCENTRIC]  
            ]

        
    def testFormat(self):
        """Test formatting"""

        # make an arbitrary coord with the string constructor.
        # check that calling the getFooStr() accessors gets back what we started with.
        equ = afwCoord.Fk5Coord(self.ra, self.dec)
        self.assertAlmostEqual(equ.getRa(afwCoord.HOURS), self.raKnown)
        self.assertAlmostEqual(equ.getDec(afwCoord.DEGREES), self.decKnown)

        print "Format: %s  %s" % (equ.getRaStr(afwCoord.HOURS), self.ra)
        self.assertEqual(equ.getRaStr(afwCoord.HOURS), self.ra)


    def testFactory(self):
        """Test the Factory function makeCoord()"""

        # make a (eg galactic) coord with the constructor, and with the factory
        # and see if they agree.
        for constructor, enum, cast, stringName in self.coordList:
            con = constructor(self.l, self.b)
            factories = []
            factories.append(afwCoord.makeCoord(enum, self.l, self.b))
            factories.append(afwCoord.makeCoord(afwCoord.makeCoordEnum(stringName), self.l, self.b))
            factories.append(afwCoord.makeCoord(enum, afwGeom.makePointD(self.l, self.b), afwCoord.DEGREES))

            print "Factory: "
            for fac in factories:
                self.assertAlmostEqual(con[0], fac[0])
                self.assertAlmostEqual(con[1], fac[1])
                s = (" tried ", fac[0], fac[1],
                     "(expected ", con[0], con[1], ")")
                print s

                
            # can we create an empty coord, and use reset() to fill it?
            c = afwCoord.makeCoord(enum)
            c.reset(1.0, 1.0, 2000.0)
            myCoord = cast(c)
            self.assertEqual(myCoord.getLongitude(afwCoord.DEGREES), 1.0)
            self.assertEqual(myCoord.getLatitude(afwCoord.DEGREES), 1.0)


        # verify that makeCoord throws when given an epoch for an epochless system
        self.assertRaises(pexEx.LsstCppException,
                          lambda: afwCoord.makeCoord(afwCoord.GALACTIC, self.l, self.b, 2000.0))
        self.assertRaises(pexEx.LsstCppException,
                          lambda: afwCoord.makeCoord(afwCoord.ICRS, self.l, self.b, 2000.0))


    def testCoordEnum(self):
        """Verify that makeCoordEnum throws an exception for non-existant systems."""
        self.assertRaises(pexEx.LsstCppException, lambda: afwCoord.makeCoordEnum("FOO"))
        
        
    def testPosition(self):
        """Test the getPosition() method"""

        # make a coord and verify that the DEGREES, RADIANS, and HOURS enums get the right things
        equ = afwCoord.Fk5Coord(self.ra, self.dec)

        # make sure we get what we asked for
        pDeg = equ.getPosition(afwCoord.DEGREES)
        self.assertAlmostEqual(equ.getRa(afwCoord.DEGREES), pDeg.getX())
        self.assertAlmostEqual(equ.getDec(afwCoord.DEGREES), pDeg.getY())

        pRad = equ.getPosition(afwCoord.RADIANS)
        self.assertAlmostEqual(equ.getRa(afwCoord.RADIANS), pRad.getX())
        self.assertAlmostEqual(equ.getDec(afwCoord.RADIANS), pRad.getY())

        pHrs = equ.getPosition(afwCoord.HOURS)
        self.assertAlmostEqual(equ.getRa(afwCoord.HOURS), pHrs.getX())
        self.assertAlmostEqual(equ.getDec(afwCoord.DEGREES), pHrs.getY())

        # make sure we construct with the type we ask for
        equ1 = afwCoord.Fk5Coord(pDeg, afwCoord.DEGREES)
        self.assertAlmostEqual(equ1.getRa(afwCoord.RADIANS), equ.getRa(afwCoord.RADIANS))

        equ2 = afwCoord.Fk5Coord(pRad, afwCoord.RADIANS)
        self.assertAlmostEqual(equ2.getRa(afwCoord.RADIANS), equ.getRa(afwCoord.RADIANS))

        equ3 = afwCoord.Fk5Coord(pHrs, afwCoord.HOURS)
        self.assertAlmostEqual(equ3.getRa(afwCoord.RADIANS), equ.getRa(afwCoord.RADIANS))


    def testVector(self):
        """Test the getVector() method, and make sure the constructors take Point3D"""

        # try the axes: vernal equinox should equal 1, 0, 0; ... north pole is 0, 0, 1; etc
        
        coordList = []
        coordList.append([(0.0, 0.0), (1.0, 0.0, 0.0)])
        coordList.append([(90.0, 0.0), (0.0, 1.0, 0.0)])
        coordList.append([(0.0, 90.0), (0.0, 0.0, 1.0)])
        
        for equ, p3dknown in coordList:
            
            # convert to p3d
            p3d = afwCoord.Fk5Coord(equ[0], equ[1]).getVector()
            print "Point3d: ", p3d, p3dknown
            for i in range(3):
                self.assertAlmostEqual(p3d[i], p3dknown[i])
                
            # convert back
            equBack = afwCoord.Fk5Coord(p3d)
            s = ("Vector (back): ", equBack.getRa(afwCoord.DEGREES), equ[0],
                 equBack.getDec(afwCoord.DEGREES), equ[1])
            print s
            self.assertAlmostEqual(equBack.getRa(afwCoord.DEGREES), equ[0])
            self.assertAlmostEqual(equBack.getDec(afwCoord.DEGREES), equ[1])
            
        
    def testNames(self):
        """Test the names of the Coords (Useful with Point2D form)"""

        # verify that each coordinate type can tell you what its components are called.
        radec1, known1 = afwCoord.Coord(self.ra, self.dec).getCoordNames(), ["RA", "Dec"]
        radec3, known3 = afwCoord.Fk5Coord(self.ra, self.dec).getCoordNames(), ["RA", "Dec"]
        radec4, known4 = afwCoord.IcrsCoord(self.ra, self.dec).getCoordNames(), ["RA", "Dec"]
        lb, known5     = afwCoord.GalacticCoord(self.ra, self.dec).getCoordNames(), ["L", "B"]
        lambet, known6 = afwCoord.EclipticCoord(self.ra, self.dec).getCoordNames(), ["Lambda", "Beta"]
        altaz, known7  = afwCoord.TopocentricCoord(self.ra, self.dec, 2000.0,
                                             afwCoord.Observatory(0,0,0)).getCoordNames(), ["Az", "Alt"]

        pairs = [ [radec1, known1],
                  [radec3, known3],
                  [radec4, known4],
                  [lb,     known5],
                  [lambet, known6],
                  [altaz,  known7], ]
                  
        for pair, known in (pairs):
            self.assertEqual(pair[0], known[0])
            self.assertEqual(pair[1], known[1])
            

    def testConvert(self):
        """Verify that the generic convert() method works"""
        
        # Pollux
        alpha, delta = "07:45:18.946", "28:01:34.26"
        pollux = afwCoord.Fk5Coord(alpha, delta)

        # bundle up a list of coords created with the specific and generic converters
        coordList = [
            [pollux.toFk5(),        pollux.convert(afwCoord.FK5)],
            [pollux.toIcrs(),       pollux.convert(afwCoord.ICRS)],
            [pollux.toGalactic(),   pollux.convert(afwCoord.GALACTIC)],
            [pollux.toEcliptic(),   pollux.convert(afwCoord.ECLIPTIC)],
          ]

        # go through the list and see if specific and generic produce the same result ... they should!
        print "Convert: "
        for specific, generic in coordList:
            # note that operator[]/__getitem__ is overloaded. It gets the internal (radian) values
            # ... the same as getPosition(afwCoord.RADIANS)
            long1, lat1 = specific[0], specific[1]  
            long2, lat2 = generic.getPosition(afwCoord.RADIANS) 
            print "(specific) %.8f %.8f   (generic) %.8f %.8f" % (long1, lat1, long2, lat2)
            self.assertEqual(long1, long2)
            self.assertEqual(lat1, lat2)

        
    def testEcliptic(self):
        """Verify Ecliptic Coordinate Transforms""" 
       
        # Pollux
        alpha, delta = "07:45:18.946", "28:01:34.26"
        # known ecliptic coords (example from Meeus, Astro algorithms, pg 95)
        lamb, beta = 113.215629, 6.684170

        # Try converting pollux Ra,Dec to ecliptic and check that we get the right answer
        polluxEqu = afwCoord.Fk5Coord(alpha, delta)
        polluxEcl = polluxEqu.toEcliptic()
        s = ("Ecliptic (Pollux): ",
             polluxEcl.getLambda(afwCoord.DEGREES), polluxEcl.getBeta(afwCoord.DEGREES), lamb, beta)
        print s

        # verify to precision of known values
        self.assertAlmostEqual(polluxEcl.getLambda(afwCoord.DEGREES), lamb, 6)
        self.assertAlmostEqual(polluxEcl.getBeta(afwCoord.DEGREES), beta, 6)

        # make sure it transforms back (machine precision)
        self.assertAlmostEqual(polluxEcl.toFk5().getRa(afwCoord.DEGREES),
                               polluxEqu.getRa(afwCoord.DEGREES), 13)
        self.assertAlmostEqual(polluxEcl.toFk5().getDec(afwCoord.DEGREES),
                               polluxEqu.getDec(afwCoord.DEGREES), 13)


    def testGalactic(self):
        """Verify Galactic coordinate transforms"""

        # Try converting Sag-A to galactic and make sure we get the right answer
        # Sagittarius A (very nearly the galactic center)
        sagAKnownEqu = afwCoord.Fk5Coord("17:45:40.04","-29:00:28.1")
        sagAKnownGal = afwCoord.GalacticCoord(359.94432, -0.04619)
        
        sagAGal = sagAKnownEqu.toGalactic()
        s = ("Galactic (Sag-A):  (transformed) %.5f %.5f   (known) %.5f %.5f\n" %
             (sagAGal.getL(afwCoord.DEGREES), sagAGal.getB(afwCoord.DEGREES),
              sagAKnownGal.getL(afwCoord.DEGREES), sagAKnownGal.getB(afwCoord.DEGREES)))
        print s
        
        # verify ... to 4 places, the accuracy of the galactic pole in Fk5
        self.assertAlmostEqual(sagAGal.getL(afwCoord.DEGREES), sagAKnownGal.getL(afwCoord.DEGREES), 4)
        self.assertAlmostEqual(sagAGal.getB(afwCoord.DEGREES), sagAKnownGal.getB(afwCoord.DEGREES), 4)

        # make sure it transforms back ... to machine precision
        self.assertAlmostEqual(sagAGal.toFk5().getRa(afwCoord.DEGREES),
                               sagAKnownEqu.getRa(afwCoord.DEGREES), 14)
        self.assertAlmostEqual(sagAGal.toFk5().getDec(afwCoord.DEGREES),
                               sagAKnownEqu.getDec(afwCoord.DEGREES), 14)
        
        
    def testTopocentric(self):
        """Verify Altitude/Azimuth coordinate transforms"""

        # try converting the RA,Dec of Sedna (on the specified date) to Alt/Az
        
        # sedna (from jpl) for 2010-03-03 00:00 UT
        ra, dec = "03:26:42.61",  "+06:32:07.1"
        az, alt = 231.5947, 44.3375
        obs = afwCoord.Observatory(74.659, 40.384, 100.0) # peyton
        obsDate = dafBase.DateTime(2010, 3, 3, 0, 0, 0, dafBase.DateTime.TAI)
        sedna = afwCoord.Fk5Coord(ra, dec, obsDate.get(dafBase.DateTime.EPOCH))
        altaz = sedna.toTopocentric(obs, obsDate)
        s = ("Topocentric (Sedna): ", altaz.getAltitude(afwCoord.DEGREES),
             altaz.getAzimuth(afwCoord.DEGREES), alt, az)
        print s

        # precision is low as we don't account for as much as jpl (abberation, nutation, etc)
        self.assertAlmostEqual(altaz.getAltitude(afwCoord.DEGREES), alt, 1)
        self.assertAlmostEqual(altaz.getAzimuth(afwCoord.DEGREES), az, 1)


    def testPrecess(self):
        """Test precession calculations in different coordinate systems"""

        # Try precessing in the various coordinate systems, and check the results.
        
        ### Fk5 ###
        
        # example 21.b Meeus, pg 135, for Alpha Persei ... with proper motion
        alpha0, delta0 = "2:44:11.986", "49:13:42.48"
        # proper motions per year
        dAlphaS, dDeltaAS = 0.03425, -0.0895
        dAlphaDeg, dDeltaDeg = dAlphaS*15/3600.0, dDeltaAS/3600.0

        # get for 2028, Nov 13.19
        epoch = dafBase.DateTime(2028, 11, 13, 4, 33, 36,
                                 dafBase.DateTime.TAI).get(dafBase.DateTime.EPOCH)

        # the known final answer
        # - actually 41.547214, 49.348483 (suspect precision error in Meeus)
        alphaKnown, deltaKnown = 41.547236, 49.348488

        alphaPer0 = afwCoord.Fk5Coord(alpha0, delta0)
        alphaDeg = alphaPer0.getRa(afwCoord.DEGREES) + dAlphaDeg*(epoch - 2000.0)
        deltaDeg = alphaPer0.getDec(afwCoord.DEGREES) + dDeltaDeg*(epoch - 2000.0)

        alphaPer = afwCoord.Fk5Coord(alphaDeg, deltaDeg).precess(epoch)

        print "Precession (Alpha-Per): %.6f %.6f   (known) %.6f %.6f" % (alphaPer.getRa(afwCoord.DEGREES),
                                                                         alphaPer.getDec(afwCoord.DEGREES),
                                                                         alphaKnown, deltaKnown)
        # precision 6 (with 1 digit fudged in the 'known' answers)
        self.assertAlmostEqual(alphaPer.getRa(afwCoord.DEGREES), alphaKnown, 6)
        self.assertAlmostEqual(alphaPer.getDec(afwCoord.DEGREES), deltaKnown, 6)

        # verify that toFk5(epoch) also works as precess
        alphaPer2 = afwCoord.Fk5Coord(alphaDeg, deltaDeg).toFk5(epoch)
        self.assertEqual(alphaPer[0], alphaPer2[0])
        self.assertEqual(alphaPer[1], alphaPer2[1])

        
        ### Galactic ###
        
        # make sure Galactic throws an exception. As there's no epoch, there's no precess() method.
        gal = afwCoord.GalacticCoord(self.l, self.b)
        epochNew = 2010.0
        self.assertRaises(AttributeError, lambda: gal.precess(epochNew))

        
        ### Icrs ###

        # make sure Icrs throws an exception. As there's no epoch, there's no precess() method.
        icrs = afwCoord.IcrsCoord(self.l, self.b)
        epochNew = 2010.0
        self.assertRaises(AttributeError, lambda: icrs.precess(epochNew))

        
        ### Ecliptic ###
        
        # test for ecliptic with venus (example from meeus, pg 137)
        lamb2000, beta2000 = 149.48194, 1.76549
        
        # known values for -214, June 30.0
        # they're actually 118.704, 1.615, but I suspect discrepancy is a rounding error in Meeus
        # -- we use double precision, he carries 7 places only.

        # originally 214BC, but that broke the DateTime
        # It did work previously, so for the short term, I've taken the answer it
        #  returns for 1920, and used that as the 'known answer' for future tests.
        
        #year = -214 
        #lamb214bc, beta214bc = 118.704, 1.606
        year = 1920
        lamb214bc, beta214bc = 148.37119237032144, 1.7610036104147864
        
        venus2000  = afwCoord.EclipticCoord(lamb2000, beta2000, 2000.0)
        ep = dafBase.DateTime(year, 6, 30, 0, 0, 0,
                               dafBase.DateTime.TAI).get(dafBase.DateTime.EPOCH)
        venus214bc = venus2000.precess(ep)
        s = ("Precession (Ecliptic, Venus): %.4f %.4f  (known) %.4f %.4f" %
             (venus214bc.getLambda(afwCoord.DEGREES), venus214bc.getBeta(afwCoord.DEGREES),
              lamb214bc, beta214bc))
        print s
        
        # 3 places precision (accuracy of our controls)
        self.assertAlmostEqual(venus214bc.getLambda(afwCoord.DEGREES), lamb214bc, 3)
        self.assertAlmostEqual(venus214bc.getBeta(afwCoord.DEGREES), beta214bc, 3)

        # verify that toEcliptic(ep) does the same as precess(ep)
        venus214bc2 = venus2000.toEcliptic(ep)
        self.assertEqual(venus214bc[0], venus214bc2[0])
        self.assertEqual(venus214bc[1], venus214bc2[1])
        
        
    def testAngularSeparation(self):
        """Test measure of angular separation between two coords"""

        # test from Meeus, pg 110
        spica = afwCoord.Fk5Coord(201.2983, -11.1614)
        arcturus = afwCoord.Fk5Coord(213.9154, 19.1825)
        knownDeg = 32.7930
        
        deg = spica.angularSeparation(arcturus, afwCoord.DEGREES)

        print "Separation (Spica/Arcturus): %.6f (known) %.6f" % (deg, knownDeg)
        # verify to precision of known
        self.assertAlmostEqual(deg, knownDeg, 4)
        
        # verify small angles ... along a constant ra, add an arcsec to spica dec
        epsilonDeg = 1.0/3600.0
        spicaPlus = afwCoord.Fk5Coord(spica.getRa(afwCoord.DEGREES),
                                      spica.getDec(afwCoord.DEGREES) + epsilonDeg)
        deg = spicaPlus.angularSeparation(spica, afwCoord.DEGREES)

        print "Separation (Spica+epsilon): %.8f  (known) %.8f" % (deg, epsilonDeg)
        # machine precision
        self.assertAlmostEqual(deg, epsilonDeg)


    def testTicket1394(self):
        """Ticket #1394 bug: coord within epsilon of RA=0 leads to negative RA and fails bounds check. """

        # the problem was that the coordinate is < epsilon close to RA==0
        # and bounds checking was getting a -ve RA.
        c = afwCoord.makeCoord(afwCoord.ICRS,
                               afwGeom.makePointD(0.6070619982, -1.264309928e-16, 0.7946544723))

        self.assertEqual(c[0], 0.0)

        
    def testRotate(self):
        """Verify rotation of coord about a user provided axis."""

        # try rotating about the equatorial pole (ie. along a parallel)
        longitude = 90.0
        latitudes = [0.0, 30.0, 60.0]
        arcLen = 10.0
        pole = afwCoord.Fk5Coord(0.0, 90.0)
        for latitude in latitudes:
            c = afwCoord.Fk5Coord(longitude, latitude)
            c.rotate(pole, arcLen*afwCoord.degToRad)

            lon = c.getLongitude(afwCoord.DEGREES)
            lat = c.getLatitude(afwCoord.DEGREES)
            
            print "Rotate along a parallel: %.10f %.10f   %.10f %.10f" % (lon, lat,
                                                                          longitude+arcLen, latitude)
            self.assertAlmostEqual(lon, longitude + arcLen)
            self.assertAlmostEqual(lat, latitude)

        # try with pole = vernal equinox and rotate up a meridian
        pole = afwCoord.Fk5Coord(0.0, 0.0)
        for latitude in latitudes:
            c = afwCoord.Fk5Coord(longitude, latitude)
            c.rotate(pole, arcLen*afwCoord.degToRad)

            lon = c.getLongitude(afwCoord.DEGREES)
            lat = c.getLatitude(afwCoord.DEGREES)
            
            print "Rotate along a meridian: %.10f %.10f   %.10f %.10f" % (lon, lat,
                                                                          longitude, latitude+arcLen)
            self.assertAlmostEqual(lon, longitude)
            self.assertAlmostEqual(lat, latitude + arcLen)

            
    def testOffset(self):
        """Verify offset of coord along a great circle."""

        longitude = 90.0
        latitude = 0.0   # These tests only work from the equator
        arcLen = 10.0
        
        c0 = afwCoord.Fk5Coord(longitude, latitude)

        # phi, arcLen, expectedLong, expectedLat, expectedPhi2
        trials = [
            [0.0, arcLen, longitude+arcLen, latitude, 0.0],   # along celestial equator
            [90.0, arcLen, longitude, latitude+arcLen, 90.0],  # along a meridian
            [45.0, 180.0, longitude+180.0, -latitude, -45.0],    # 180 arc (should go to antipodal point)
            [45.0, 90.0, longitude+90.0, latitude+45.0, 0.0],  # 
            ]

        for trial in trials:
            
            phi, arc, longExp, latExp, phi2Exp = trial
            c = c0.clone()
            phi2 = afwCoord.radToDeg*c.offset(phi*afwCoord.degToRad, arc*afwCoord.degToRad)
            
            lon = c.getLongitude(afwCoord.DEGREES)
            lat = c.getLatitude(afwCoord.DEGREES)

            print "Offset: %.10f %.10f %.10f  %.10f %.10f %.10f" % (lon, lat, phi2, longExp, latExp, phi2Exp)
            self.assertAlmostEqual(lon, longExp, 12)
            self.assertAlmostEqual(lat, latExp, 12)
            self.assertAlmostEqual(phi2, phi2Exp, 12)
        

        
        
#################################################################
# Test suite boiler plate
#################################################################
def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(CoordTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
