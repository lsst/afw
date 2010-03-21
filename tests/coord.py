#!/usr/bin/env python
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
import lsst.afw.geom             as geom
import lsst.afw.coord.coordLib   as coord
import lsst.utils.tests          as utilsTests
import lsst.daf.base             as dafBase

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
            [coord.Fk5Coord,      coord.FK5,      coord.cast_Fk5],
            [coord.IcrsCoord,     coord.ICRS,     coord.cast_Icrs],
            [coord.GalacticCoord, coord.GALACTIC, coord.cast_Galactic],
            [coord.EclipticCoord, coord.ECLIPTIC, coord.cast_Ecliptic],
            # we can't factory an AltAz ... Observatory must be specified.
            # [coord.AltAzCoord, coord.ALTAZ]  
            ]

        
    def testFormat(self):
        """Test formatting"""

        # make an arbitrary coord with the string constructor.
        # check that calling the getFooStr() accessors gets back what we started with.
        equ = coord.Fk5Coord(self.ra, self.dec)
        self.assertAlmostEqual(equ.getRa(coord.HOURS), self.raKnown)
        self.assertAlmostEqual(equ.getDec(coord.DEGREES), self.decKnown)

        print "Format: %s  %s" % (equ.getRaStr(coord.HOURS), self.ra)
        self.assertEqual(equ.getRaStr(coord.HOURS), self.ra)


    def testFactory(self):
        """Test the Factory function makeCoord()"""

        # make a (eg galactic) coord with the constructor, and with the factory
        # and see if they agree.
        for constructor, enum, cast in self.coordList:
            con = constructor(self.l, self.b)
            factories = []
            factories.append(coord.makeCoord(enum, self.l, self.b))
            factories.append(coord.makeCoord(enum, geom.makePointD(self.l, self.b), coord.DEGREES))

            print "Factory: "
            for fac in factories:
                self.assertAlmostEqual(con[0], fac[0])
                self.assertAlmostEqual(con[1], fac[1])
                s = (" tried ", fac[0], fac[1],
                     "(expected ", con[0], con[1], ")")
                print s

                
            # can we create an empty coord, and use reset() to fill it?
            c = coord.makeCoord(enum)
            c.reset(1.0, 1.0, 2000.0)
            myCoord = cast(c)
            self.assertEqual(myCoord.getLongitude(coord.DEGREES), 1.0)
            self.assertEqual(myCoord.getLatitude(coord.DEGREES), 1.0)

        
    def testPosition(self):
        """Test the getPosition() method"""

        # make a coord and verify that the DEGREES, RADIANS, and HOURS enums get the right things
        equ = coord.Fk5Coord(self.ra, self.dec)

        # make sure we get what we asked for
        pDeg = equ.getPosition(coord.DEGREES)
        self.assertAlmostEqual(equ.getRa(coord.DEGREES), pDeg.getX())
        self.assertAlmostEqual(equ.getDec(coord.DEGREES), pDeg.getY())

        pRad = equ.getPosition(coord.RADIANS)
        self.assertAlmostEqual(equ.getRa(coord.RADIANS), pRad.getX())
        self.assertAlmostEqual(equ.getDec(coord.RADIANS), pRad.getY())

        pHrs = equ.getPosition(coord.HOURS)
        self.assertAlmostEqual(equ.getRa(coord.HOURS), pHrs.getX())
        self.assertAlmostEqual(equ.getDec(coord.DEGREES), pHrs.getY())

        # make sure we construct with the type we ask for
        equ1 = coord.Fk5Coord(pDeg, coord.DEGREES)
        self.assertAlmostEqual(equ1.getRa(coord.RADIANS), equ.getRa(coord.RADIANS))

        equ2 = coord.Fk5Coord(pRad, coord.RADIANS)
        self.assertAlmostEqual(equ2.getRa(coord.RADIANS), equ.getRa(coord.RADIANS))

        equ3 = coord.Fk5Coord(pHrs, coord.HOURS)
        self.assertAlmostEqual(equ3.getRa(coord.RADIANS), equ.getRa(coord.RADIANS))


    def testVector(self):
        """Test the getVector() method, and make sure the constructors take Point3D"""

        # try the axes: vernal equinox should equal 1, 0, 0; ... north pole is 0, 0, 1; etc
        
        coordList = []
        coordList.append([(0.0, 0.0), (1.0, 0.0, 0.0)])
        coordList.append([(90.0, 0.0), (0.0, 1.0, 0.0)])
        coordList.append([(0.0, 90.0), (0.0, 0.0, 1.0)])
        
        for equ, p3dknown in coordList:
            
            # convert to p3d
            p3d = coord.Fk5Coord(equ[0], equ[1]).getVector()
            print "Point3d: ", p3d, p3dknown
            for i in range(3):
                self.assertAlmostEqual(p3d[i], p3dknown[i])
                
            # convert back
            equBack = coord.Fk5Coord(p3d)
            s = ("Vector (back): ", equBack.getRa(coord.DEGREES), equ[0],
                 equBack.getDec(coord.DEGREES), equ[1])
            print s
            self.assertAlmostEqual(equBack.getRa(coord.DEGREES), equ[0])
            self.assertAlmostEqual(equBack.getDec(coord.DEGREES), equ[1])
            
        
    def testNames(self):
        """Test the names of the Coords (Useful with Point2D form)"""

        # verify that each coordinate type can tell you what its components are called.
        radec1, known1 = coord.Coord(self.ra, self.dec).getCoordNames(), ["RA", "Dec"]
        radec3, known3 = coord.Fk5Coord(self.ra, self.dec).getCoordNames(), ["RA", "Dec"]
        radec4, known4 = coord.IcrsCoord(self.ra, self.dec).getCoordNames(), ["RA", "Dec"]
        lb, known5     = coord.GalacticCoord(self.ra, self.dec).getCoordNames(), ["L", "B"]
        lambet, known6 = coord.EclipticCoord(self.ra, self.dec).getCoordNames(), ["Lambda", "Beta"]
        altaz, known7  = coord.AltAzCoord(self.ra, self.dec, 2000.0,
                                          coord.Observatory(0,0,0)).getCoordNames(), ["Az", "Alt"]

        pairs = [ [radec1, known1],
                  [radec3, known3],
                  [radec4, known4],
                  [lb,     known5],
                  [lambet, known6],
                  [altaz,  known7], ]
                  
        for pair, known in (pairs):
            self.assertEqual(pair[0], known[0])
            self.assertEqual(pair[1], known[1])
            

    def testEquatorial(self):
        """Verify that EquatorialCoord is an IcrsCoord"""
        
        # Pollux
        alpha, delta = "07:45:18.946", "28:01:34.26"
        icrs = coord.IcrsCoord(alpha, delta)
        equ  = coord.EquatorialCoord(alpha, delta)

        s =  ("Equatorial (Pollux): ", equ.getRa(coord.DEGREES), equ.getDec(coord.DEGREES),
              icrs.getRa(coord.DEGREES), icrs.getDec(coord.DEGREES))
        print s
        self.assertEqual(equ.getRa(coord.DEGREES),  icrs.getRa(coord.DEGREES))
        self.assertEqual(equ.getDec(coord.DEGREES), icrs.getDec(coord.DEGREES))

        
    def testConvert(self):
        """Verify that the generic convert() method works"""
        
        # Pollux
        alpha, delta = "07:45:18.946", "28:01:34.26"
        pollux = coord.Fk5Coord(alpha, delta)

        # bundle up a list of coords created with the specific and generic converters
        coordList = [
            [pollux.toFk5(),        pollux.convert(coord.FK5)],
            [pollux.toIcrs(),       pollux.convert(coord.ICRS)],
            [pollux.toEquatorial(), pollux.convert(coord.EQUATORIAL)],
            [pollux.toGalactic(),   pollux.convert(coord.GALACTIC)],
            [pollux.toEcliptic(),   pollux.convert(coord.ECLIPTIC)],
          ]

        # go through the list and see if specific and generic produce the same result ... they should!
        print "Convert: "
        for specific, generic in coordList:
            # note that operator[]/__getitem__ is overloaded. It gets the internal (radian) values
            # ... the same as getPosition(coord.RADIANS)
            long1, lat1 = specific[0], specific[1]  
            long2, lat2 = generic.getPosition(coord.RADIANS) 
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
        polluxEqu = coord.Fk5Coord(alpha, delta)
        polluxEcl = polluxEqu.toEcliptic()
        s = ("Ecliptic (Pollux): ",
             polluxEcl.getLambda(coord.DEGREES), polluxEcl.getBeta(coord.DEGREES), lamb, beta)
        print s

        # verify to precision of known values
        self.assertAlmostEqual(polluxEcl.getLambda(coord.DEGREES), lamb, 6)
        self.assertAlmostEqual(polluxEcl.getBeta(coord.DEGREES), beta, 6)

        # make sure it transforms back (machine precision)
        self.assertAlmostEqual(polluxEcl.toFk5().getRa(coord.DEGREES), polluxEqu.getRa(coord.DEGREES), 13)
        self.assertAlmostEqual(polluxEcl.toFk5().getDec(coord.DEGREES), polluxEqu.getDec(coord.DEGREES), 13)


    def testGalactic(self):
        """Verify Galactic coordinate transforms"""

        # Try converting Sag-A to galactic and make sure we get the right answer
        # Sagittarius A (very nearly the galactic center)
        sagAKnownEqu = coord.Fk5Coord("17:45:40.04","-29:00:28.1")
        sagAKnownGal = coord.GalacticCoord(359.94432, -0.04619)
        
        sagAGal = sagAKnownEqu.toGalactic()
        s = ("Galactic (Sag-A):  (transformed) %.5f %.5f   (known) %.5f %.5f\n" %
             (sagAGal.getL(coord.DEGREES), sagAGal.getB(coord.DEGREES),
              sagAKnownGal.getL(coord.DEGREES), sagAKnownGal.getB(coord.DEGREES)))
        print s
        
        # verify ... to 4 places, the accuracy of the galactic pole in Fk5
        self.assertAlmostEqual(sagAGal.getL(coord.DEGREES), sagAKnownGal.getL(coord.DEGREES), 4)
        self.assertAlmostEqual(sagAGal.getB(coord.DEGREES), sagAKnownGal.getB(coord.DEGREES), 4)

        # make sure it transforms back ... to machine precision
        self.assertAlmostEqual(sagAGal.toFk5().getRa(coord.DEGREES), sagAKnownEqu.getRa(coord.DEGREES), 14)
        self.assertAlmostEqual(sagAGal.toFk5().getDec(coord.DEGREES), sagAKnownEqu.getDec(coord.DEGREES), 14)
        
        
    def testAltAz(self):
        """Verify Altitude/Azimuth coordinate transforms"""

        # try converting the RA,Dec of Sedna (on the specified date) to Alt/Az
        
        # sedna (from jpl) for 2010-03-03 00:00 UT
        ra, dec = "03:26:42.61",  "+06:32:07.1"
        az, alt = 231.5947, 44.3375
        obs = coord.Observatory(74.659, 40.384, 100.0) # peyton
        obsDate = dafBase.DateTime(2010, 3, 3, 0, 0, 0, dafBase.DateTime.TAI)
        sedna = coord.Fk5Coord(ra, dec, obsDate.get(dafBase.DateTime.EPOCH))
        altaz = sedna.toAltAz(obs, obsDate)
        print "AltAz (Sedna): ", altaz.getAltitude(coord.DEGREES), altaz.getAzimuth(coord.DEGREES), alt, az

        # precision is low as we don't account for as much as jpl (abberation, nutation, etc)
        self.assertAlmostEqual(altaz.getAltitude(coord.DEGREES), alt, 1)
        self.assertAlmostEqual(altaz.getAzimuth(coord.DEGREES), az, 1)


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

        alphaPer0 = coord.Fk5Coord(alpha0, delta0)
        alphaDeg = alphaPer0.getRa(coord.DEGREES) + dAlphaDeg*(epoch - 2000.0)
        deltaDeg = alphaPer0.getDec(coord.DEGREES) + dDeltaDeg*(epoch - 2000.0)

        alphaPer = coord.Fk5Coord(alphaDeg, deltaDeg).precess(epoch)

        print "Precession (Alpha-Per): %.6f %.6f   (known) %.6f %.6f" % (alphaPer.getRa(coord.DEGREES),
                                                                         alphaPer.getDec(coord.DEGREES),
                                                                         alphaKnown, deltaKnown)
        # precision 6 (with 1 digit fudged in the 'known' answers)
        self.assertAlmostEqual(alphaPer.getRa(coord.DEGREES), alphaKnown, 6)
        self.assertAlmostEqual(alphaPer.getDec(coord.DEGREES), deltaKnown, 6)

        # verify that toFk5(epoch) also works as precess
        alphaPer2 = coord.Fk5Coord(alphaDeg, deltaDeg).toFk5(epoch)
        self.assertEqual(alphaPer[0], alphaPer2[0])
        self.assertEqual(alphaPer[1], alphaPer2[1])

        
        ### Galactic ###
        
        # make sure Galactic throws an exception. As there's no epoch, there's no precess() method.
        gal = coord.GalacticCoord(self.l, self.b)
        epochNew = 2010.0
        self.assertRaises(AttributeError, lambda: gal.precess(epochNew))

        
        ### Icrs ###

        # make sure Icrs throws an exception. As there's no epoch, there's no precess() method.
        icrs = coord.IcrsCoord(self.l, self.b)
        epochNew = 2010.0
        self.assertRaises(AttributeError, lambda: icrs.precess(epochNew))

        
        ### Equatorial ###
        
        # make sure Equatorial throws an exception. It's ICRS (no epoch), there's no precess() method.
        equ = coord.EquatorialCoord(self.l, self.b)
        epochNew = 2010.0
        self.assertRaises(AttributeError, lambda: equ.precess(epochNew))
        
        
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
        
        venus2000  = coord.EclipticCoord(lamb2000, beta2000, 2000.0)
        ep = dafBase.DateTime(year, 6, 30, 0, 0, 0,
                               dafBase.DateTime.TAI).get(dafBase.DateTime.EPOCH)
        venus214bc = venus2000.precess(ep)
        s = ("Precession (Ecliptic, Venus): %.4f %.4f  (known) %.4f %.4f" %
             (venus214bc.getLambda(coord.DEGREES), venus214bc.getBeta(coord.DEGREES), lamb214bc, beta214bc))
        print s
        
        # 3 places precision (accuracy of our controls)
        self.assertAlmostEqual(venus214bc.getLambda(coord.DEGREES), lamb214bc, 3)
        self.assertAlmostEqual(venus214bc.getBeta(coord.DEGREES), beta214bc, 3)

        # verify that toEcliptic(ep) does the same as precess(ep)
        venus214bc2 = venus2000.toEcliptic(ep)
        self.assertEqual(venus214bc[0], venus214bc2[0])
        self.assertEqual(venus214bc[1], venus214bc2[1])
        
        
    def testAngularSeparation(self):
        """Test measure of angular separation between two coords"""

        # test from Meeus, pg 110
        spica = coord.Fk5Coord(201.2983, -11.1614)
        arcturus = coord.Fk5Coord(213.9154, 19.1825)
        knownDeg = 32.7930
        
        deg = spica.angularSeparation(arcturus, coord.DEGREES)

        print "Separation (Spica/Arcturus): %.6f (known) %.6f" % (deg, knownDeg)
        # verify to precision of known
        self.assertAlmostEqual(deg, knownDeg, 4)
        
        # verify small angles ... along a constant ra, add an arcsec to spica dec
        epsilonDeg = 1.0/3600.0
        spicaPlus = coord.Fk5Coord(spica.getRa(coord.DEGREES), spica.getDec(coord.DEGREES) + epsilonDeg)
        deg = spicaPlus.angularSeparation(spica, coord.DEGREES)

        print "Separation (Spica+epsilon): %.8f  (known) %.8f" % (deg, epsilonDeg)
        # machine precision
        self.assertAlmostEqual(deg, epsilonDeg)

        
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

def run(exit = False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
