#!/usr/bin/env python
from __future__ import absolute_import, division
from __future__ import print_function
from builtins import zip
from builtins import range

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

"""
Tests for Calib, Color, and Filter

Run with:
   color.py
or
   python
   >>> import color; color.run()
"""

import math
import unittest

import numpy

import lsst.utils.tests as tests
import lsst.daf.base as dafBase
import lsst.pex.logging as logging
import lsst.pex.exceptions as pexExcept
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.image.utils as imageUtils
from lsst.afw.cameraGeom.testUtils import DetectorWrapper

try:
    type(verbose)
except NameError:
    verbose = 0
    logging.Debug("afwDetect.Footprint", verbose)

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class CalibTestCase(unittest.TestCase):
    """A test case for Calib"""

    def setUp(self):
        self.calib = afwImage.Calib()
        self.detector = DetectorWrapper().detector

    def tearDown(self):
        del self.calib
        del self.detector

    def testTime(self):
        """Test the exposure time information"""

        isoDate = "1995-01-26T07:32:00.000000000Z"
        self.calib.setMidTime(dafBase.DateTime(isoDate))
        self.assertEqual(isoDate, self.calib.getMidTime().toString())
        self.assertAlmostEqual(self.calib.getMidTime().get(), 49743.3142245)

        dt = 123.4
        self.calib.setExptime(dt)
        self.assertEqual(self.calib.getExptime(), dt)

    def testDetectorTime(self):
        """Test that we can ask a calib for the MidTime at a point in a detector (ticket #1337)"""
        p = afwGeom.PointI(3, 4)
        self.calib.getMidTime(self.detector, p)

    def testPhotom(self):
        """Test the zero-point information"""

        flux, fluxErr = 1000.0, 10.0
        flux0, flux0Err = 1e12, 1e10
        self.calib.setFluxMag0(flux0)

        self.assertEqual(flux0, self.calib.getFluxMag0()[0])
        self.assertEqual(0.0, self.calib.getFluxMag0()[1])
        self.assertEqual(22.5, self.calib.getMagnitude(flux))
        # Error just in flux
        self.assertAlmostEqual(self.calib.getMagnitude(flux, fluxErr)[1], 2.5/math.log(10)*fluxErr/flux)
        # Error just in flux0
        self.calib.setFluxMag0(flux0, flux0Err)
        self.assertEqual(flux0Err, self.calib.getFluxMag0()[1])
        self.assertAlmostEqual(self.calib.getMagnitude(flux, 0)[1], 2.5/math.log(10)*flux0Err/flux0)

        self.assertAlmostEqual(flux0, self.calib.getFlux(0))
        self.assertAlmostEqual(flux, self.calib.getFlux(22.5))

        # I don't know how to test round-trip if fluxMag0 is significant compared to fluxErr
        self.calib.setFluxMag0(flux0, flux0 / 1e6)
        for fluxErr in (flux / 1e2, flux / 1e4):
            mag, magErr = self.calib.getMagnitude(flux, fluxErr)
            self.assertAlmostEqual(flux, self.calib.getFlux(mag, magErr)[0])
            self.assertTrue(abs(fluxErr - self.calib.getFlux(mag, magErr)[1]) < 1.0e-4)

        # Test context manager; shouldn't raise an exception within the block, should outside
        with imageUtils.CalibNoThrow():
            self.assert_(numpy.isnan(self.calib.getMagnitude(-50.0)))
        self.assertRaises(pexExcept.DomainError, self.calib.getMagnitude, -50.0)

    def testPhotomMulti(self):
        self.calib.setFluxMag0(1e12, 1e10)
        flux, fluxErr = 1000.0, 10.0
        num = 5

        mag, magErr = self.calib.getMagnitude(flux, fluxErr)  # Result assumed to be true: tested elsewhere

        fluxList = numpy.array([flux for i in range(num)], dtype=float)
        fluxErrList = numpy.array([fluxErr for i in range(num)], dtype=float)

        magList = self.calib.getMagnitude(fluxList)
        for m in magList:
            self.assertEqual(m, mag)

        mags, magErrs = self.calib.getMagnitude(fluxList, fluxErrList)

        for m, dm in zip(mags, magErrs):
            self.assertEqual(m, mag)
            self.assertEqual(dm, magErr)

        flux, fluxErr = self.calib.getFlux(mag, magErr)  # Result assumed to be true: tested elsewhere

        fluxList = self.calib.getFlux(magList)
        for f in fluxList:
            self.assertEqual(f, flux)

        fluxes = self.calib.getFlux(mags, magErrs)
        for f, df in zip(fluxes[0], fluxes[1]):
            self.assertAlmostEqual(f, flux)
            self.assertAlmostEqual(df, fluxErr)

    def testCtorFromMetadata(self):
        """Test building a Calib from metadata"""

        isoDate = "1995-01-26T07:32:00.000000000Z"
        exptime = 123.4
        flux0, flux0Err = 1e12, 1e10
        flux, fluxErr = 1000.0, 10.0

        metadata = dafBase.PropertySet()
        metadata.add("TIME-MID", isoDate)
        metadata.add("EXPTIME", exptime)
        metadata.add("FLUXMAG0", flux0)
        metadata.add("FLUXMAG0ERR", flux0Err)

        self.calib = afwImage.Calib(metadata)

        self.assertEqual(isoDate, self.calib.getMidTime().toString())
        self.assertAlmostEqual(self.calib.getMidTime().get(), 49743.3142245)
        self.assertEqual(self.calib.getExptime(), exptime)

        self.assertEqual(flux0, self.calib.getFluxMag0()[0])
        self.assertEqual(flux0Err, self.calib.getFluxMag0()[1])
        self.assertEqual(22.5, self.calib.getMagnitude(flux))
        # Error just in flux
        self.calib.setFluxMag0(flux0, 0)

        self.assertAlmostEqual(self.calib.getMagnitude(flux, fluxErr)[1], 2.5/math.log(10)*fluxErr/flux)

        #
        # Check that we can clean up metadata
        #
        afwImage.stripCalibKeywords(metadata)
        self.assertEqual(len(metadata.names()), 0)

    def testCalibEquality(self):
        self.assertEqual(self.calib, self.calib)
        self.assertFalse(self.calib != self.calib)

        calib2 = afwImage.Calib()
        calib2.setExptime(12)

        self.assertNotEqual(calib2, self.calib)

    def testCalibFromCalibs(self):
        """Test creating a Calib from an array of Calibs"""
        exptime = 20
        mag0, mag0Sigma = 1.0, 0.01
        time0 = dafBase.DateTime.now().get()

        calibs = afwImage.vectorCalib()
        ncalib = 3
        for i in range(ncalib):
            calib = afwImage.Calib()
            calib.setMidTime(dafBase.DateTime(time0 + i))
            calib.setExptime(exptime)
            calib.setFluxMag0(mag0, mag0Sigma)

            calibs.append(calib)

        ocalib = afwImage.Calib(calibs)

        self.assertEqual(ocalib.getExptime(), ncalib*exptime)
        self.assertAlmostEqual(calibs[ncalib//2].getMidTime().get(), ocalib.getMidTime().get())
        #
        # Check that we can only merge Calibs with the same fluxMag0 values
        #
        calibs[0].setFluxMag0(1.001*mag0, mag0Sigma)
        self.assertRaises(pexExcept.InvalidParameterError,
                          lambda: afwImage.Calib(calibs))

    def testCalibNegativeFlux(self):
        """Check that we can control if -ve fluxes raise exceptions"""
        self.calib.setFluxMag0(1e12)

        funcs = [lambda: self.calib.getMagnitude(-10), lambda: self.calib.getMagnitude(-10, 1)]

        for func in funcs:
            self.assertRaises(pexExcept.DomainError, func)

        afwImage.Calib.setThrowOnNegativeFlux(False)
        for func in funcs:
            mags = func()
            try:                        # deal with returning mag or [mag, magErr]
                mags[0]
            except TypeError:
                mags = [mags, None]

            for m in mags:
                if m is not None:
                    self.assertTrue(numpy.isnan(m))

        afwImage.Calib.setThrowOnNegativeFlux(True)

        for func in funcs:
            self.assertRaises(pexExcept.DomainError, func)


def defineSdssFilters(testCase):
    # Initialise filters as used for our tests
    imageUtils.resetFilters()
    wavelengths = dict()
    testCase.aliases = dict(u=[], g=[], r=[], i=[], z=['zprime', "z'"])
    for name, lambdaEff in (('u', 355.1), ('g', 468.6), ('r', 616.5), ('i', 748.1), ('z', 893.1)):
        wavelengths[name] = lambdaEff
        imageUtils.defineFilter(name, lambdaEff, alias=testCase.aliases[name])
    return wavelengths


class ColorTestCase(unittest.TestCase):
    """A test case for Color"""

    def setUp(self):
        defineSdssFilters(self)

    def tearDown(self):
        pass

    def testCtor(self):
        afwImage.Color()
        afwImage.Color(1.2)

    def testLambdaEff(self):
        f = afwImage.Filter("g")
        g_r = 1.2
        c = afwImage.Color(g_r)

        self.assertEqual(c.getLambdaEff(f), 1000*g_r)  # XXX Not a real implementation!

    def testIsIndeterminate(self):
        """Test that a default-constructed Color tests True, but ones with a g-r value test False"""
        self.assertTrue(afwImage.Color().isIndeterminate())
        self.assertFalse(afwImage.Color(1.2).isIndeterminate())


class FilterTestCase(unittest.TestCase):
    """A test case for Filter"""

    def setUp(self):
        # Initialise our filters
        #
        # Start by forgetting that we may already have defined filters
        #
        wavelengths = defineSdssFilters(self)
        self.filters = tuple(sorted(wavelengths.keys()))
        self.g_lambdaEff = [lambdaEff for name,
                            lambdaEff in wavelengths.items() if name == "g"][0]  # for tests

    def defineFilterProperty(self, name, lambdaEff, force=False):
        return afwImage.FilterProperty(name, lambdaEff, force)

    def testListFilters(self):
        self.assertEqual(afwImage.Filter_getNames(), self.filters)

    def testCtor(self):
        """Test that we can construct a Filter"""
        # A filter of type
        afwImage.Filter("g")

    def testCtorFromMetadata(self):
        """Test building a Filter from metadata"""

        metadata = dafBase.PropertySet()
        metadata.add("FILTER", "g")

        f = afwImage.Filter(metadata)
        self.assertEqual(f.getName(), "g")
        #
        # Check that we can clean up metadata
        #
        afwImage.stripFilterKeywords(metadata)
        self.assertEqual(len(metadata.names()), 0)

        badFilter = "rhl"               # an undefined filter
        metadata.add("FILTER", badFilter)
        # Not defined
        self.assertRaises(pexExcept.NotFoundError,
                          lambda: afwImage.Filter(metadata))
        # Force definition
        f = afwImage.Filter(metadata, True)
        self.assertEqual(f.getName(), badFilter)  # name is correctly defined

    def testFilterEquality(self):
        # a "g" filter
        f = afwImage.Filter("g")
        g = afwImage.Filter("g")

        self.assertEqual(f, g)

        f = afwImage.Filter()           # the unknown filter
        self.assertNotEqual(f, f)       # ... doesn't equal itself

    def testFilterProperty(self):
        # a "g" filter
        f = afwImage.Filter("g")
        # The properties of a g filter
        g = afwImage.FilterProperty.lookup("g")

        if False:
            print("Filter: %s == %d lambda_{eff}=%g" % (f.getName(), f.getId(),
                                                        f.getFilterProperty().getLambdaEff()))

        self.assertEqual(f.getName(), "g")
        self.assertEqual(f.getId(), 1)
        self.assertEqual(f.getFilterProperty().getLambdaEff(), self.g_lambdaEff)
        self.assertTrue(f.getFilterProperty() ==
                        self.defineFilterProperty("gX", self.g_lambdaEff, True))

        self.assertEqual(g.getLambdaEff(), self.g_lambdaEff)

    def testFilterAliases(self):
        """Test that we can provide an alias for a Filter"""
        for name0 in self.aliases:
            f0 = afwImage.Filter(name0)
            self.assertEqual(f0.getCanonicalName(), name0)
            self.assertEqual(sorted(f0.getAliases()), sorted(self.aliases[name0]))

            for name in self.aliases[name0]:
                f = afwImage.Filter(name)
                self.assertEqual(sorted(f.getAliases()), sorted(self.aliases[name0]))
                self.assertEqual(f.getId(), f0.getId())
                self.assertEqual(f.getName(), name)
                self.assertEqual(afwImage.Filter(f.getId()).getName(), name0)
                self.assertEqual(f.getCanonicalName(), name0)
                self.assertNotEqual(f.getCanonicalName(), name)
                self.assertEqual(f.getFilterProperty().getLambdaEff(), f0.getFilterProperty().getLambdaEff())

    def testReset(self):
        """Test that we can reset filter IDs and properties if needs be"""
        # The properties of a g filter
        g = afwImage.FilterProperty.lookup("g")
        #
        # First FilterProperty
        #

        def tst():
            self.defineFilterProperty("g", self.g_lambdaEff + 10)

        self.assertRaises(pexExcept.RuntimeError, tst)
        self.defineFilterProperty("g", self.g_lambdaEff + 10, True)  # should not raise
        self.defineFilterProperty("g", self.g_lambdaEff, True)
        #
        # Can redefine
        #

        def tst():
            self.defineFilterProperty("g", self.g_lambdaEff + 10)  # changing definition is not allowed
        self.assertRaises(pexExcept.RuntimeError, tst)

        self.defineFilterProperty("g", self.g_lambdaEff)  # identical redefinition is allowed
        #
        # Now Filter
        #
        afwImage.Filter.define(g, afwImage.Filter("g").getId())  # OK if Id's the same
        afwImage.Filter.define(g, afwImage.Filter.AUTO)         # AUTO will assign the same ID

        def tst():
            afwImage.Filter.define(g, afwImage.Filter("g").getId() + 10)  # different ID

        self.assertRaises(pexExcept.RuntimeError, tst)

    def testUnknownFilter(self):
        """Test that we can define, but not use, an unknown filter"""
        badFilter = "rhl"               # an undefined filter
        # Not defined
        self.assertRaises(pexExcept.NotFoundError,
                          lambda: afwImage.Filter(badFilter))
        # Force definition
        f = afwImage.Filter(badFilter, True)
        self.assertEqual(f.getName(), badFilter)  # name is correctly defined

        self.assertRaises(pexExcept.NotFoundError,
                          lambda: f.getFilterProperty().getLambdaEff())  # can't use Filter f
        #
        # Now define badFilter
        #
        lambdaEff = 666.0
        self.defineFilterProperty(badFilter, lambdaEff)

        self.assertEqual(f.getFilterProperty().getLambdaEff(), lambdaEff)  # but now we can
        #
        # Check that we didn't accidently define the unknown filter
        #
        self.assertRaises(pexExcept.NotFoundError,
                          lambda: afwImage.Filter().getFilterProperty().getLambdaEff())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(CalibTestCase)
    suites += unittest.makeSuite(ColorTestCase)
    suites += unittest.makeSuite(FilterTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests"""
    tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
