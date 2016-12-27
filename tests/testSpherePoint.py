#
# LSST Data Management System
# See COPYRIGHT file at the top of the source tree.
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
Unit tests for SpherePoint

Run with:
   python testSpherePoint.py
or
   python
   >>> import testSpherePoint
   >>> testSpherePoint.run()
"""

from __future__ import absolute_import, division, print_function

import copy
import math
import re
import unittest

import numpy as np

import lsst.utils.tests
import lsst.afw.geom as afwGeom
import lsst.pex.exceptions as pexEx

from lsst.afw.geom import degrees, radians, SpherePoint
from numpy import nan, inf


class SpherePointTestSuite(lsst.utils.tests.TestCase):

    def setUp(self):
        self._dataset = SpherePointTestSuite.positions()
        self._poleLatitudes = [
            afwGeom.HALFPI*afwGeom.radians,
            6.0*afwGeom.hours,
            90.0*afwGeom.degrees,
            5400.0*afwGeom.arcminutes,
            324000.0*afwGeom.arcseconds,
        ]

    @staticmethod
    def positions():
        """Provide valid coordinates for nominal-case testing.

        Returns
        -------
        positions : `iterable`
            An iterable of pairs of Angles, each representing the
            longitude and latitude (in that order) of a test point.
        """
        nValidPoints = 100
        rng = np.random.RandomState(42)
        ra = rng.uniform(0.0, 360.0, nValidPoints)
        dec = rng.uniform(-90.0, 90.0, nValidPoints)

        points = list(zip(ra*degrees, dec*degrees))
        # Ensure corner cases are tested.
        points += [
            (0.0*degrees, 0.0*degrees),
            (afwGeom.PI*radians, -6.0*degrees),
            (42.0*degrees, -90.0*degrees),
            (172.0*degrees, afwGeom.HALFPI*radians),
            (360.0*degrees, 45.0*degrees),
            (-278.0*degrees, -42.0*degrees),
            (765.0*degrees, 0.25*afwGeom.PI*radians),
            (180.0*degrees, nan*radians),
            (inf*degrees, 45.0*degrees),
            (nan*degrees, -8.3*degrees),
        ]
        return points

    def testInit2ArgErrors(self):
        """Test if 2-argument form of __init__ handles invalid input.
        """
        # Latitude should be checked for out-of-range.
        for lat in self._poleLatitudes:
            with self.assertRaises(pexEx.OutOfRangeError):
                SpherePoint(0.0*degrees, self.nextUp(lat))
            with self.assertRaises(pexEx.OutOfRangeError):
                SpherePoint(0.0*degrees, self.nextDown(0.0*radians - lat))

        # Longitude should not be checked for out of range.
        SpherePoint(360.0*degrees, 45.0*degrees)
        SpherePoint(-42.0*degrees, 45.0*degrees)
        SpherePoint(391.0*degrees, 45.0*degrees)

        # Infinite latitude is not allowed.
        with self.assertRaises(pexEx.OutOfRangeError):
            SpherePoint(-42.0*degrees, inf*degrees)
        with self.assertRaises(pexEx.OutOfRangeError):
            SpherePoint(-42.0*degrees, -inf*degrees)

    def testInit1ArgErrors(self):
        """Test if 1-argument form of __init__ handles invalid input.
        """
        # Only one singularity, at zero
        with self.assertRaises(pexEx.InvalidParameterError):
            SpherePoint(afwGeom.Point3D(0.0, 0.0, 0.0))
        SpherePoint(afwGeom.Point3D(0.0, -0.2, 0.0))
        SpherePoint(afwGeom.Point3D(0.0, 0.0, 1.0))
        SpherePoint(afwGeom.Point3D(42.78, -46.29, 38.27))

    def testCopyConstructor(self):
        sp = SpherePoint(-42.0*degrees, 45.0*degrees)
        spcopy = SpherePoint(sp)
        self.assertEqual(sp, spcopy)

    def testInitNArgFail(self):
        """Tests if only 1- or 2-argument initializers are allowed.
        """
        with self.assertRaises(TypeError):
            SpherePoint()
        with self.assertRaises(TypeError):
            SpherePoint("Rotund", "Bovine")
        with self.assertRaises(TypeError):
            SpherePoint(42)
        with self.assertRaises(TypeError):
            SpherePoint("ICRS", 34.0, -56.0)

    def testGetLongitudeValue(self):
        """Test if getLongitude() returns the expected value.
        """
        for lon, lat in self._dataset:
            point = SpherePoint(lon, lat)
            self.assertIsInstance(point.getLongitude(), afwGeom.Angle)
            # Behavior for non-finite points is undefined; depends on internal data representation
            if point.isFinite():
                self.assertGreaterEqual(point.getLongitude().asDegrees(), 0.0)
                self.assertLess(point.getLongitude().asDegrees(), 360.0)

                # Longitude not guaranteed to match input at pole
                if not point.atPole():
                    # assertAnglesNearlyEqual handles angle wrapping internally
                    self.assertAnglesNearlyEqual(lon, point.getLongitude())

        # Vector construction should return valid longitude even in edge cases.
        point = SpherePoint(afwGeom.Point3D(0.0, 0.0, -1.0))
        self.assertGreaterEqual(point.getLongitude().asDegrees(), 0.0)
        self.assertLess(point.getLongitude().asDegrees(), 360.0)

    def testTicket1394(self):
        """Regression test for Ticket 1761.

        Checks that negative longitudes within epsilon of lon=0 lead
        are correctly bounded and rounded.
        """
        # The problem was that the coordinate is less than epsilon
        # close to RA == 0 and bounds checking was getting a
        # negative RA.
        point = SpherePoint(afwGeom.Point3D(0.6070619982, -1.264309928e-16, 0.7946544723))

        self.assertEqual(point[0].asDegrees(), 0.0)

    def testGetLatitudeValue(self):
        """Test if getLatitude() returns the expected value.
        """
        for lon, lat in self._dataset:
            point = SpherePoint(lon, lat)
            self.assertIsInstance(point.getLatitude(), afwGeom.Angle)
            # Behavior for non-finite points is undefined; depends on internal data representation
            if point.isFinite():
                self.assertGreaterEqual(point.getLatitude().asDegrees(), -90.0)
                self.assertLessEqual(point.getLatitude().asDegrees(), 90.0)
                self.assertAnglesNearlyEqual(lat, point.getLatitude())

    def testGetVectorValue(self):
        """Test if getVector() returns the expected value.

        The test includes conformance to vector-angle conventions.
        """
        pointList = [
            ((0.0, 0.0), afwGeom.Point3D(1.0, 0.0, 0.0)),
            ((90.0, 0.0), afwGeom.Point3D(0.0, 1.0, 0.0)),
            ((0.0, 90.0), afwGeom.Point3D(0.0, 0.0, 1.0)),
        ]

        for lonLat, vector in pointList:
            # Convert to Point3D.
            point = SpherePoint(lonLat[0]*degrees, lonLat[1]*degrees)
            newVector = point.getVector()
            self.assertIsInstance(newVector, afwGeom.Point3D)
            for oldElement, newElement in zip(vector, newVector):
                self.assertAlmostEqual(oldElement, newElement)

            # Convert back to spherical.
            newLon, newLat = SpherePoint(newVector)
            self.assertAlmostEqual(newLon.asDegrees(), lonLat[0])
            self.assertAlmostEqual(newLat.asDegrees(), lonLat[1])

        # Try some un-normalized ones, too.
        pointList = [
            ((0.0, 0.0), afwGeom.Point3D(1.3, 0.0, 0.0)),
            ((90.0, 0.0), afwGeom.Point3D(0.0, 1.2, 0.0)),
            ((0.0, 90.0), afwGeom.Point3D(0.0, 0.0, 2.3)),
            ((0.0, 0.0), afwGeom.Point3D(0.5, 0.0, 0.0)),
            ((90.0, 0.0), afwGeom.Point3D(0.0, 0.7, 0.0)),
            ((0.0, 90.0), afwGeom.Point3D(0.0, 0.0, 0.9)),
        ]

        for lonLat, vector in pointList:
            # Only convert from vector to spherical.
            point = SpherePoint(vector)
            newLon, newLat = point
            self.assertAlmostEqual(lonLat[0], newLon.asDegrees())
            self.assertAlmostEqual(lonLat[1], newLat.asDegrees())
            self.assertAlmostEqual(1.0, point.getVector().distanceSquared(afwGeom.Point3D(0.0, 0.0, 0.0)))

        # Ill-defined points should be all NaN after normalization
        cleanVector = afwGeom.Point3D(0.5, -0.3, 0.2)
        badValues = [nan, inf, -inf]
        for i in range(3):
            for badValue in badValues:
                # Ensure each subtest is independent
                dirtyVector = copy.deepcopy(cleanVector)
                dirtyVector[i] = badValue
                for element in SpherePoint(dirtyVector).getVector():
                    self.assertTrue(math.isnan(element))

    def testTicket1761(self):
        """Regression test for Ticket 1761.

        Checks for math errors caused by unnormalized vectors.
        """
        refPoint = SpherePoint(afwGeom.Point3D(0, 1, 0))

        point1 = SpherePoint(afwGeom.Point3D(0.1, 0.1, 0.1))
        point2 = SpherePoint(afwGeom.Point3D(0.6, 0.6, 0.6))
        sep1 = refPoint.separation(point1)
        sep2 = refPoint.separation(point2)
        sepTrue = 54.735610317245339*degrees

        self.assertAnglesNearlyEqual(sepTrue, sep1)
        self.assertAnglesNearlyEqual(sepTrue, sep2)

    def testAtPoleValue(self):
        """Test if atPole() returns the expected value.
        """
        poleList = [SpherePoint(42.0*degrees, lat) for lat in self._poleLatitudes] + \
            [SpherePoint(42.0*degrees, 0.0*radians - lat) for lat in self._poleLatitudes] + [
            SpherePoint(afwGeom.Point3D(0.0, 0.0, 1.0)),
            SpherePoint(afwGeom.Point3D(0.0, 0.0, -1.0)),
        ]
        nonPoleList = [SpherePoint(42.0*degrees, self.nextDown(lat)) for lat in self._poleLatitudes] + \
            [SpherePoint(42.0*degrees, self.nextUp(0.0*radians - lat)) for lat in self._poleLatitudes] + [
            SpherePoint(afwGeom.Point3D(9.9e-7, 0.0, 1.0)),
            SpherePoint(afwGeom.Point3D(9.9e-7, 0.0, -1.0)),
            SpherePoint(0.0*degrees, nan*degrees),
        ]

        for pole in poleList:
            self.assertIsInstance(pole.atPole(), bool)
            self.assertTrue(pole.atPole())

        for nonPole in nonPoleList:
            self.assertIsInstance(nonPole.atPole(), bool)
            self.assertFalse(nonPole.atPole())

    def testIsFiniteValue(self):
        """Test if isFinite() returns the expected value.
        """
        finiteList = [
            SpherePoint(0.0*degrees, -90.0*degrees),
            SpherePoint(afwGeom.Point3D(0.1, 0.2, 0.3)),
        ]
        nonFiniteList = [
            SpherePoint(0.0*degrees, nan*degrees),
            SpherePoint(nan*degrees, 0.0*degrees),
            SpherePoint(inf*degrees, 0.0*degrees),
            SpherePoint(-inf*degrees, 0.0*degrees),
            SpherePoint(afwGeom.Point3D(nan, 0.2, 0.3)),
            SpherePoint(afwGeom.Point3D(0.1, inf, 0.3)),
            SpherePoint(afwGeom.Point3D(0.1, 0.2, -inf)),
        ]

        for finite in finiteList:
            self.assertIsInstance(finite.isFinite(), bool)
            self.assertTrue(finite.isFinite())

        for nonFinite in nonFiniteList:
            self.assertIsInstance(nonFinite.isFinite(), bool)
            self.assertFalse(nonFinite.isFinite())

    def testGetItemError(self):
        """Test if indexing correctly handles invalid input.
        """
        point = SpherePoint(afwGeom.Point3D(1.0, 1.0, 1.0))

        with self.assertRaises(pexEx.OutOfRangeError):
            point[2]
        with self.assertRaises(pexEx.OutOfRangeError):
            point[-3]

    def testGetItemValue(self):
        """Test if indexing returns the expected value.
        """
        for lon, lat in self._dataset:
            point = SpherePoint(lon, lat)
            self.assertIsInstance(point[-2], afwGeom.Angle)
            self.assertIsInstance(point[-1], afwGeom.Angle)
            self.assertIsInstance(point[0], afwGeom.Angle)
            self.assertIsInstance(point[1], afwGeom.Angle)

            if not math.isnan(point.getLongitude().asRadians()):
                self.assertEqual(point.getLongitude(), point[-2])
                self.assertEqual(point.getLongitude(), point[0])
            else:
                self.assertTrue(math.isnan(point[-2].asRadians()))
                self.assertTrue(math.isnan(point[0].asRadians()))
            if not math.isnan(point.getLatitude().asRadians()):
                self.assertEqual(point.getLatitude(), point[-1])
                self.assertEqual(point.getLatitude(), point[1])
            else:
                self.assertTrue(math.isnan(point[-1].asRadians()))
                self.assertTrue(math.isnan(point[1].asRadians()))

    def testEquality(self):
        """Test if tests for equality treat SpherePoints as values.
        """
        # (In)equality is determined by value, not identity.
        # See DM-2347, DM-2465. These asserts are testing the
        # functionality of `==` and `!=` and should not be changed.
        for lon1, lat1 in self._dataset:
            point1 = SpherePoint(lon1, lat1)
            self.assertIsInstance(point1 == point1, bool)
            self.assertIsInstance(point1 != point1, bool)
            if point1.isFinite():
                self.assertTrue(point1 == point1)
                self.assertFalse(point1 != point1)

                pointCopy = copy.deepcopy(point1)
                self.assertIsNot(pointCopy, point1)
                self.assertEqual(pointCopy, point1)
                self.assertEqual(point1, pointCopy)
                self.assertFalse(pointCopy != point1)
                self.assertFalse(point1 != pointCopy)
            else:
                self.assertFalse(point1 == point1)
                self.assertTrue(point1 != point1)

            for lon2, lat2 in self._dataset:
                if lon1 == lon2 and lat1 == lat2:
                    continue
                point2 = SpherePoint(lon2, lat2)
                self.assertTrue(point2 != point1)
                self.assertTrue(point1 != point2)
                self.assertFalse(point2 == point1)
                self.assertFalse(point1 == point2)

        # Test for transitivity (may be assumed by algorithms).
        for delta in [10.0**(0.1*x) for x in range(-150, -49, 5)]:
            self.checkTransitive(delta*radians)

    def checkTransitive(self, delta):
        """Test if equality is transitive even for close points.

        This test prevents misuse of approximate floating-point
        equality -- if `__eq__` is implemented using AFP, then this
        test will fail for some value of `delta`. Testing multiple
        values is recommended.

        Parameters
        ----------
        delta : `number`
            The separation, in degrees, at which point equality may
            become intransitive.
        """
        for lon, lat in self._dataset:
            point1 = SpherePoint(lon - delta, lat)
            point2 = SpherePoint(lon, lat)
            point3 = SpherePoint(lon + delta, lat)

            self.assertTrue(point1 != point2 or point2 != point3 or point1 == point3)
            self.assertTrue(point3 != point1 or point1 != point2 or point3 == point2)
            self.assertTrue(point2 == point3 or point3 != point1 or point2 == point1)

    def testEqualityAlias(self):
        """Test if == handles coordinate degeneracies correctly.
        """
        # Longitude wrapping
        self.assertEqual(SpherePoint(360.0*degrees, -42.0*degrees),
                         SpherePoint(0.0*degrees, -42.0*degrees))
        self.assertEqual(SpherePoint(-90.0*degrees, -42.0*degrees),
                         SpherePoint(270.0*degrees, -42.0*degrees))

        # Polar degeneracy
        self.assertEqual(SpherePoint(42.0*degrees, 90.0*degrees),
                         SpherePoint(270.0*degrees, 90.0*degrees))
        self.assertEqual(SpherePoint(-42.0*degrees, -90.0*degrees),
                         SpherePoint(83.0*degrees, -90.0*degrees))

        self.assertNotEqual(SpherePoint(83.0*degrees, 90.0*degrees),
                            SpherePoint(83.0*degrees, -90.0*degrees))
        # White-box test: how are pole/non-pole comparisons handled?
        self.assertNotEqual(SpherePoint(42.0*degrees, 90.0*degrees),
                            SpherePoint(42.0*degrees, 0.0*degrees))

    def testInquality(self):
        """Test if == and != give mutually consistent results.
        """
        for lon1, lat1 in self._dataset:
            point1 = SpherePoint(lon1, lat1)
            for lon2, lat2 in self._dataset:
                point2 = SpherePoint(lon2, lat2)
                self.assertEqual((point1 == point2), (point2 == point1))
                self.assertEqual((point1 != point2), (point2 != point1))
                self.assertNotEqual((point1 == point2), (point1 != point2))

    def testBearingToError(self):
        """Test if bearingTo() correctly handles invalid input.
        """
        northPole = SpherePoint(0.0*degrees, 90.0*degrees)
        southPole = SpherePoint(0.0*degrees, -90.0*degrees)
        safe = SpherePoint(0.0*degrees, 0.0*degrees)

        with self.assertRaises(pexEx.DomainError):
            northPole.bearingTo(safe)
        with self.assertRaises(pexEx.DomainError):
            southPole.bearingTo(safe)

    def testBearingToValue(self):
        """Test if bearingTo() returns the expected value.
        """
        lon0 = 90.0
        lat0 = 0.0   # These tests only work from the equator.
        arcLen = 10.0

        trials = [
            # Along celestial equator
            dict(lon=lon0, lat=lat0, bearing=0.0, lonEnd=lon0+arcLen, latEnd=lat0),
            # Along a meridian
            dict(lon=lon0, lat=lat0, bearing=90.0, lonEnd=lon0, latEnd=lat0+arcLen),
            # 180 degree arc (should go to antipodal point)
            dict(lon=lon0, lat=lat0, bearing=45.0, lonEnd=lon0+180.0, latEnd=-lat0),
            #
            dict(lon=lon0, lat=lat0, bearing=45.0, lonEnd=lon0+90.0, latEnd=lat0 + 45.0),
            dict(lon=lon0, lat=lat0, bearing=225.0, lonEnd=lon0-90.0, latEnd=lat0 - 45.0),
            dict(lon=lon0, lat=np.nextafter(-90.0, inf), bearing=90.0, lonEnd=lon0, latEnd=0.0),
            dict(lon=lon0, lat=np.nextafter(-90.0, inf), bearing=0.0, lonEnd=lon0 + 90.0, latEnd=0.0),
            # Argument at a pole should work
            dict(lon=lon0, lat=lat0, bearing=270.0, lonEnd=lon0, latEnd=-90.0),
            # Support for non-finite values
            dict(lon=lon0, lat=nan, bearing=nan, lonEnd=lon0, latEnd=45.0),
            dict(lon=lon0, lat=lat0, bearing=nan, lonEnd=nan, latEnd=90.0),
            dict(lon=inf, lat=lat0, bearing=nan, lonEnd=lon0, latEnd=42.0),
            dict(lon=lon0, lat=lat0, bearing=nan, lonEnd=-inf, latEnd=42.0),
        ]

        for trial in trials:
            origin = SpherePoint(trial['lon']*degrees, trial['lat']*degrees)
            end = SpherePoint(trial['lonEnd']*degrees, trial['latEnd']*degrees)
            bearing = origin.bearingTo(end)

            self.assertIsInstance(bearing, afwGeom.Angle)
            if origin.isFinite() and end.isFinite():
                self.assertGreaterEqual(bearing.asDegrees(), 0.0)
                self.assertLess(bearing.asDegrees(), 360.0)
            if origin.separation(end).asDegrees() != 180.0:
                if not math.isnan(trial['bearing']):
                    self.assertAlmostEqual(trial['bearing'], bearing.asDegrees(), 12)
                else:
                    self.assertTrue(math.isnan(bearing.asRadians()))

    def testBearingToValueSingular(self):
        """White-box test: bearingTo() may be unstable if points are near opposite poles.

        This test is motivated by an error analysis of the `bearingTo`
        implementation. It may become irrelevant if the implementation
        changes.
        """
        southPole = SpherePoint(0.0*degrees, self.nextUp(-90.0*degrees))
        northPoleSame = SpherePoint(0.0*degrees, self.nextDown(90.0*degrees))
        # Don't let it be on exactly the opposite side.
        northPoleOpposite = SpherePoint(180.0*degrees, self.nextDown(northPoleSame.getLatitude()))

        self.assertAnglesNearlyEqual(southPole.bearingTo(northPoleSame), afwGeom.HALFPI*afwGeom.radians)
        self.assertAnglesNearlyEqual(southPole.bearingTo(northPoleOpposite),
                                     (afwGeom.PI + afwGeom.HALFPI)*afwGeom.radians)

    def testSeparationValueGeneric(self):
        """Test if separation() returns the correct value.
        """
        # This should cover arcs over the meridian, across the pole, etc.
        # Do not use sphgeom as an oracle, in case SpherePoint uses it
        # internally.
        for lon1, lat1 in self._dataset:
            point1 = SpherePoint(lon1, lat1)
            x1, y1, z1 = SpherePointTestSuite.toVector(lon1, lat1)
            for lon2, lat2 in self._dataset:
                point2 = SpherePoint(lon2, lat2)
                if lon1 != lon2 or lat1 != lat2:
                    # Numerically unstable at small angles, but that's ok.
                    x2, y2, z2 = SpherePointTestSuite.toVector(lon2, lat2)
                    expected = math.acos(x1*x2 + y1*y2 + z1*z2)
                else:
                    expected = 0.0

                sep = point1.separation(point2)
                self.assertIsInstance(sep, afwGeom.Angle)
                if point1.isFinite() and point2.isFinite():
                    self.assertGreaterEqual(sep.asDegrees(), 0.0)
                    self.assertLessEqual(sep.asDegrees(), 180.0)
                    self.assertAlmostEqual(expected, sep.asRadians())
                    self.assertAnglesNearlyEqual(sep, point2.separation(point1))
                else:
                    self.assertTrue(math.isnan(sep.asRadians()))
                    self.assertTrue(math.isnan(point2.separation(point1).asRadians()))

    def testSeparationValueAbsolute(self):
        """Test if separation() returns specific values.
        """
        # Test from "Meeus, p. 110" (test originally written for coord::Coord; don't know exact reference)
        spica = SpherePoint(201.2983*degrees, -11.1614*degrees)
        arcturus = SpherePoint(213.9154*degrees, 19.1825*degrees)

        # Verify to precision of quoted distance and positions.
        self.assertAlmostEqual(32.7930, spica.separation(arcturus).asDegrees(), 4)

        # Verify small angles: along a constant ra, add an arcsec to spica dec.
        epsilon = 1.0*afwGeom.arcseconds
        spicaPlus = SpherePoint(spica.getLongitude(), spica.getLatitude() + epsilon)

        self.assertAnglesNearlyEqual(epsilon, spicaPlus.separation(spica))

    def testSeparationPoles(self):
        """White-box test: all representations of a pole should have the same distance to another point.
        """
        southPole1 = SpherePoint(-30.0*degrees, -90.0*degrees)
        southPole2 = SpherePoint(183.0*degrees, -90.0*degrees)
        regularPoint = SpherePoint(42.0*degrees, 45.0*degrees)
        expectedSep = (45.0+90.0)*degrees

        self.assertAnglesNearlyEqual(expectedSep, southPole1.separation(regularPoint))
        self.assertAnglesNearlyEqual(expectedSep, regularPoint.separation(southPole1))
        self.assertAnglesNearlyEqual(expectedSep, southPole2.separation(regularPoint))
        self.assertAnglesNearlyEqual(expectedSep, regularPoint.separation(southPole2))

    @staticmethod
    def toVector(longitude, latitude):
        """Converts a set of spherical coordinates to a 3-vector.

        The conversion shall not be performed by any library, to ensure
        that the test case does not duplicate the code being tested.

        Parameters
        ----------
        longitude : `Angle`
            The longitude (right ascension, azimuth, etc.) of the
            position.
        latitude : `Angle`
            The latitude (declination, elevation, etc.) of the
            position.

        Returns
        -------
        x, y, z : `number`
            Components of the unit vector representation of
            `(longitude, latitude)`
        """
        alpha = longitude.asRadians()
        delta = latitude.asRadians()
        if math.isnan(alpha) or math.isinf(alpha) or math.isnan(delta) or math.isinf(delta):
            return (nan, nan, nan)

        x = math.cos(alpha)*math.cos(delta)
        y = math.sin(alpha)*math.cos(delta)
        z = math.sin(delta)
        return (x, y, z)

    def testRotatedValue(self):
        """Test if rotated() returns the expected value.
        """
        # Try rotating about the equatorial pole (ie. along a parallel).
        longitude = 90.0
        latitudes = [0.0, 30.0, 60.0]
        arcLen = 10.0
        pole = SpherePoint(0.0*degrees, 90.0*degrees)
        for latitude in latitudes:
            point = SpherePoint(longitude*degrees, latitude*degrees)
            newPoint = point.rotated(pole, arcLen*degrees)

            self.assertIsInstance(newPoint, SpherePoint)
            self.assertAlmostEqual(longitude + arcLen, newPoint.getLongitude().asDegrees())
            self.assertAlmostEqual(latitude, newPoint.getLatitude().asDegrees())

        # Try with pole = vernal equinox and rotate up the 90 degree meridian.
        pole = SpherePoint(0.0*degrees, 0.0*degrees)
        for latitude in latitudes:
            point = SpherePoint(longitude*degrees, latitude*degrees)
            newPoint = point.rotated(pole, arcLen*degrees)

            self.assertAlmostEqual(longitude, newPoint.getLongitude().asDegrees())
            self.assertAlmostEqual(latitude + arcLen, newPoint.getLatitude().asDegrees())

        # Test accuracy close to coordinate pole
        point = SpherePoint(90.0*degrees, np.nextafter(90.0, -inf)*degrees)
        newPoint = point.rotated(pole, 90.0*degrees)
        self.assertAlmostEqual(270.0, newPoint.getLongitude().asDegrees())
        self.assertAlmostEqual(90.0 - np.nextafter(90.0, -inf), newPoint.getLatitude().asDegrees())

        # Generic pole; can't predict position, but test for rotation invariant.
        pole = SpherePoint(283.5*degrees, -23.6*degrees)
        for lon, lat in self._dataset:
            point = SpherePoint(lon, lat)
            dist = point.separation(pole)
            newPoint = point.rotated(pole, -32.4*afwGeom.radians)

            self.assertNotAlmostEqual(point.getLongitude().asDegrees(), newPoint.getLongitude().asDegrees())
            self.assertNotAlmostEqual(point.getLatitude().asDegrees(), newPoint.getLatitude().asDegrees())
            self.assertAnglesNearlyEqual(dist, newPoint.separation(pole))

        # Non-finite values give undefined rotations
        for latitude in latitudes:
            point = SpherePoint(longitude*degrees, latitude*degrees)
            nanPoint = point.rotated(pole, nan*degrees)
            infPoint = point.rotated(pole, inf*degrees)

            self.assertTrue(math.isnan(nanPoint.getLongitude().asRadians()))
            self.assertTrue(math.isnan(nanPoint.getLatitude().asRadians()))
            self.assertTrue(math.isnan(infPoint.getLongitude().asRadians()))
            self.assertTrue(math.isnan(infPoint.getLatitude().asRadians()))

        # Non-finite points rotate into non-finite points
        for point in [
            SpherePoint(-inf*degrees, 1.0*radians),
            SpherePoint(32.0*degrees, nan*radians),
        ]:
            newPoint = point.rotated(pole, arcLen*degrees)
            self.assertTrue(math.isnan(nanPoint.getLongitude().asRadians()))
            self.assertTrue(math.isnan(nanPoint.getLatitude().asRadians()))
            self.assertTrue(math.isnan(infPoint.getLongitude().asRadians()))
            self.assertTrue(math.isnan(infPoint.getLatitude().asRadians()))

        # Rotation around non-finite poles undefined
        for latitude in latitudes:
            point = SpherePoint(longitude*degrees, latitude*degrees)
            for pole in [
                SpherePoint(-inf*degrees, 1.0*radians),
                SpherePoint(32.0*degrees, nan*radians),
            ]:
                newPoint = point.rotated(pole, arcLen*degrees)
                self.assertTrue(math.isnan(nanPoint.getLongitude().asRadians()))
                self.assertTrue(math.isnan(nanPoint.getLatitude().asRadians()))
                self.assertTrue(math.isnan(infPoint.getLongitude().asRadians()))
                self.assertTrue(math.isnan(infPoint.getLatitude().asRadians()))

    def testRotatedAlias(self):
        """White-box test: all representations of a pole should rotate into the same point.
        """
        longitudes = [0.0, 90.0, 242.0]
        latitude = 90.0
        arcLen = 10.0
        pole = SpherePoint(90.0*degrees, 0.0*degrees)
        for longitude in longitudes:
            point = SpherePoint(longitude*degrees, latitude*degrees)
            newPoint = point.rotated(pole, arcLen*degrees)

            self.assertAlmostEqual(0.0, newPoint.getLongitude().asDegrees())
            self.assertAlmostEqual(80.0, newPoint.getLatitude().asDegrees())

    def testOffsetError(self):
        """Test if offset() correctly handles invalid input.
        """
        northPole = SpherePoint(0.0*degrees, 90.0*degrees)
        southPole = SpherePoint(0.0*degrees, -90.0*degrees)

        with self.assertRaises(pexEx.DomainError):
            northPole.offset(-90.0*degrees, 10.0*degrees)
        with self.assertRaises(pexEx.DomainError):
            southPole.offset(90.0*degrees, 0.1*degrees)

    def testOffsetValue(self):
        """Test if offset() returns the expected value.
        """
        # This should cover arcs over the meridian, across the pole, etc.
        for lon1, lat1 in self._dataset:
            point1 = SpherePoint(lon1, lat1)
            if point1.atPole():
                continue
            for lon2, lat2 in self._dataset:
                if lon1 == lon2 and lat1 == lat2:
                    continue
                point2 = SpherePoint(lon2, lat2)
                bearing = point1.bearingTo(point2)
                distance = point1.separation(point2)

                newPoint = point1.offset(bearing, distance)
                self.assertIsInstance(newPoint, SpherePoint)
                if point1.isFinite() and point2.isFinite():
                    if not point2.atPole():
                        self.assertAnglesNearlyEqual(point2.getLongitude(), newPoint.getLongitude())
                    self.assertAnglesNearlyEqual(point2.getLatitude(), newPoint.getLatitude())
                else:
                    self.assertTrue(math.isnan(newPoint.getLongitude().asRadians()))
                    self.assertTrue(math.isnan(newPoint.getLatitude().asRadians()))

        # Test precision near the poles
        lon = 123.0*degrees
        almostPole = SpherePoint(lon, self.nextDown(90.0*degrees))
        goSouth = almostPole.offset(-90.0*degrees, 90.0*degrees)
        self.assertAnglesNearlyEqual(lon, goSouth.getLongitude())
        self.assertAnglesNearlyEqual(0.0*degrees, goSouth.getLatitude())
        goEast = almostPole.offset(0.0*degrees, 90.0*degrees)
        self.assertAnglesNearlyEqual(lon + 90.0*degrees, goEast.getLongitude())
        self.assertAnglesNearlyEqual(0.0*degrees, goEast.getLatitude())

    def testIterResult(self):
        """Test if iteration returns the expected values.
        """
        for lon, lat in self._dataset:
            point = SpherePoint(lon, lat)
            if point.isFinite():
                # Test mechanics directly
                it = iter(point)
                self.assertEqual(point.getLongitude(), next(it))
                self.assertEqual(point.getLatitude(), next(it))
                with self.assertRaises(StopIteration):
                    next(it)

                # Intended use case
                lon, lat = point
                self.assertEqual(point.getLongitude(), lon)
                self.assertEqual(point.getLatitude(), lat)

    def testStrValue(self):
        """Test if __str__ produces output consistent with its spec.

        This is necessarily a loose test, as the behavior of __str__
        is (deliberately) incompletely specified.
        """
        for lon, lat in self._dataset:
            point = SpherePoint(lon, lat)
            numbers = re.findall(r'(?:\+|-)?(?:[\d.]+|nan|inf)', str(point))
            self.assertEqual(2, len(numbers),
                             "String '%s' should have exactly two coordinates." % (point,))

            # Low precision to allow for only a few digits in string.
            if not math.isnan(point.getLongitude().asRadians()):
                self.assertAlmostEqual(point.getLongitude().asDegrees(), float(numbers[0]), delta=1e-6)
            else:
                self.assertRegexpMatches(numbers[0], r'-?nan')
            if not math.isnan(point.getLatitude().asRadians()):
                self.assertAlmostEqual(point.getLatitude().asDegrees(), float(numbers[1]), delta=1e-6)
                # Latitude must be signed
                self.assertTrue(numbers[1].startswith("+") or numbers[1].startswith("-"))
            else:
                # Some C++ compilers will output NaN with a sign, others won't
                self.assertRegexpMatches(numbers[1], r'(?:\+|-)?nan')

    def testReprValue(self):
        """Test if __repr__ is a machine-readable representation.
        """
        for lon, lat in self._dataset:
            point = SpherePoint(lon, lat)
            pointRepr = repr(point)
            self.assertIn("degrees", pointRepr)
            self.assertEqual(2, len(pointRepr.split(",")))

            spcopy = eval(pointRepr)
            self.assertAnglesNearlyEqual(point.getLongitude(), spcopy.getLongitude())
            self.assertAnglesNearlyEqual(point.getLatitude(), spcopy.getLatitude())

    def nextUp(self, angle):
        """Returns the smallest angle that is larger than the argument.
        """
        return np.nextafter(angle.asRadians(), inf)*radians

    def nextDown(self, angle):
        """Returns the largest angle that is smaller than the argument.
        """
        return np.nextafter(angle.asRadians(), -inf)*radians


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
