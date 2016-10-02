#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import zip
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
#pybind11#import math
#pybind11#import unittest
#pybind11#
#pybind11#import numpy as np
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.daf.base as dafBase
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.image.utils as imageUtils
#pybind11#from lsst.afw.cameraGeom.testUtils import DetectorWrapper
#pybind11#
#pybind11#import lsstDebug
#pybind11#if lsstDebug.Info(__name__).verbose:
#pybind11#    logging.Debug("afwDetect.Footprint", True)
#pybind11#
#pybind11## Set to True to display things in ds9.
#pybind11#display = False
#pybind11#
#pybind11#
#pybind11#class CalibTestCase(lsst.utils.tests.TestCase):
#pybind11#    def setUp(self):
#pybind11#        self.calib = afwImage.Calib()
#pybind11#        self.detector = DetectorWrapper().detector
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.calib
#pybind11#        del self.detector
#pybind11#
#pybind11#    def testTime(self):
#pybind11#        """Test the exposure time information"""
#pybind11#
#pybind11#        isoDate = "1995-01-26T07:32:00.000000000Z"
#pybind11#        self.calib.setMidTime(dafBase.DateTime(isoDate, dafBase.DateTime.UTC))
#pybind11#        self.assertEqual(isoDate, self.calib.getMidTime().toString(dafBase.DateTime.UTC))
#pybind11#        self.assertAlmostEqual(self.calib.getMidTime().get(), 49743.3142245)
#pybind11#
#pybind11#        dt = 123.4
#pybind11#        self.calib.setExptime(dt)
#pybind11#        self.assertEqual(self.calib.getExptime(), dt)
#pybind11#
#pybind11#    def testDetectorTime(self):
#pybind11#        """Test that we can ask a calib for the MidTime at a point in a detector (ticket #1337)"""
#pybind11#        p = afwGeom.PointI(3, 4)
#pybind11#        self.calib.getMidTime(self.detector, p)
#pybind11#
#pybind11#    def testPhotom(self):
#pybind11#        """Test the zero-point information"""
#pybind11#
#pybind11#        flux, fluxErr = 1000.0, 10.0
#pybind11#        flux0, flux0Err = 1e12, 1e10
#pybind11#        self.calib.setFluxMag0(flux0)
#pybind11#
#pybind11#        self.assertEqual(flux0, self.calib.getFluxMag0()[0])
#pybind11#        self.assertEqual(0.0, self.calib.getFluxMag0()[1])
#pybind11#        self.assertEqual(22.5, self.calib.getMagnitude(flux))
#pybind11#        # Error just in flux
#pybind11#        self.assertAlmostEqual(self.calib.getMagnitude(flux, fluxErr)[1], 2.5/math.log(10)*fluxErr/flux)
#pybind11#        # Error just in flux0
#pybind11#        self.calib.setFluxMag0(flux0, flux0Err)
#pybind11#        self.assertEqual(flux0Err, self.calib.getFluxMag0()[1])
#pybind11#        self.assertAlmostEqual(self.calib.getMagnitude(flux, 0)[1], 2.5/math.log(10)*flux0Err/flux0)
#pybind11#
#pybind11#        self.assertAlmostEqual(flux0, self.calib.getFlux(0))
#pybind11#        self.assertAlmostEqual(flux, self.calib.getFlux(22.5))
#pybind11#
#pybind11#        # I don't know how to test round-trip if fluxMag0 is significant compared to fluxErr
#pybind11#        self.calib.setFluxMag0(flux0, flux0 / 1e6)
#pybind11#        for fluxErr in (flux / 1e2, flux / 1e4):
#pybind11#            mag, magErr = self.calib.getMagnitude(flux, fluxErr)
#pybind11#            self.assertAlmostEqual(flux, self.calib.getFlux(mag, magErr)[0])
#pybind11#            self.assertLess(abs(fluxErr - self.calib.getFlux(mag, magErr)[1]), 1.0e-4)
#pybind11#
#pybind11#        # Test context manager; shouldn't raise an exception within the block, should outside
#pybind11#        with imageUtils.CalibNoThrow():
#pybind11#            self.assertTrue(np.isnan(self.calib.getMagnitude(-50.0)))
#pybind11#        with self.assertRaises(pexExcept.DomainError):
#pybind11#            self.calib.getMagnitude(-50.0)
#pybind11#
#pybind11#    def testPhotomMulti(self):
#pybind11#        self.calib.setFluxMag0(1e12, 1e10)
#pybind11#        flux, fluxErr = 1000.0, 10.0
#pybind11#        num = 5
#pybind11#
#pybind11#        mag, magErr = self.calib.getMagnitude(flux, fluxErr)  # Result assumed to be true: tested elsewhere
#pybind11#
#pybind11#        fluxList = np.array([flux for i in range(num)], dtype=float)
#pybind11#        fluxErrList = np.array([fluxErr for i in range(num)], dtype=float)
#pybind11#
#pybind11#        magList = self.calib.getMagnitude(fluxList)
#pybind11#        for m in magList:
#pybind11#            self.assertEqual(m, mag)
#pybind11#
#pybind11#        mags, magErrs = self.calib.getMagnitude(fluxList, fluxErrList)
#pybind11#
#pybind11#        for m, dm in zip(mags, magErrs):
#pybind11#            self.assertEqual(m, mag)
#pybind11#            self.assertEqual(dm, magErr)
#pybind11#
#pybind11#        flux, fluxErr = self.calib.getFlux(mag, magErr)  # Result assumed to be true: tested elsewhere
#pybind11#
#pybind11#        fluxList = self.calib.getFlux(magList)
#pybind11#        for f in fluxList:
#pybind11#            self.assertEqual(f, flux)
#pybind11#
#pybind11#        fluxes = self.calib.getFlux(mags, magErrs)
#pybind11#        for f, df in zip(fluxes[0], fluxes[1]):
#pybind11#            self.assertAlmostEqual(f, flux)
#pybind11#            self.assertAlmostEqual(df, fluxErr)
#pybind11#
#pybind11#    def testCtorFromMetadata(self):
#pybind11#        """Test building a Calib from metadata"""
#pybind11#
#pybind11#        isoDate = "1995-01-26T07:32:00.000000000Z"
#pybind11#        exptime = 123.4
#pybind11#        flux0, flux0Err = 1e12, 1e10
#pybind11#        flux, fluxErr = 1000.0, 10.0
#pybind11#
#pybind11#        metadata = dafBase.PropertySet()
#pybind11#        metadata.add("TIME-MID", isoDate)
#pybind11#        metadata.add("EXPTIME", exptime)
#pybind11#        metadata.add("FLUXMAG0", flux0)
#pybind11#        metadata.add("FLUXMAG0ERR", flux0Err)
#pybind11#
#pybind11#        self.calib = afwImage.Calib(metadata)
#pybind11#
#pybind11#        self.assertEqual(isoDate, self.calib.getMidTime().toString(dafBase.DateTime.UTC))
#pybind11#        self.assertAlmostEqual(self.calib.getMidTime().get(), 49743.3142245)
#pybind11#        self.assertEqual(self.calib.getExptime(), exptime)
#pybind11#
#pybind11#        self.assertEqual(flux0, self.calib.getFluxMag0()[0])
#pybind11#        self.assertEqual(flux0Err, self.calib.getFluxMag0()[1])
#pybind11#        self.assertEqual(22.5, self.calib.getMagnitude(flux))
#pybind11#        # Error just in flux
#pybind11#        self.calib.setFluxMag0(flux0, 0)
#pybind11#
#pybind11#        self.assertAlmostEqual(self.calib.getMagnitude(flux, fluxErr)[1], 2.5/math.log(10)*fluxErr/flux)
#pybind11#
#pybind11#        # Check that we can clean up metadata
#pybind11#        afwImage.stripCalibKeywords(metadata)
#pybind11#        self.assertEqual(len(metadata.names()), 0)
#pybind11#
#pybind11#    def testCalibEquality(self):
#pybind11#        self.assertEqual(self.calib, self.calib)
#pybind11#        self.assertFalse(self.calib != self.calib)  # using assertFalse to directly test != operator
#pybind11#
#pybind11#        calib2 = afwImage.Calib()
#pybind11#        calib2.setExptime(12)
#pybind11#
#pybind11#        self.assertNotEqual(calib2, self.calib)
#pybind11#
#pybind11#    def testCalibFromCalibs(self):
#pybind11#        """Test creating a Calib from an array of Calibs"""
#pybind11#        exptime = 20
#pybind11#        mag0, mag0Sigma = 1.0, 0.01
#pybind11#        time0 = dafBase.DateTime.now().get()
#pybind11#
#pybind11#        calibs = afwImage.vectorCalib()
#pybind11#        ncalib = 3
#pybind11#        for i in range(ncalib):
#pybind11#            calib = afwImage.Calib()
#pybind11#            calib.setMidTime(dafBase.DateTime(time0 + i))
#pybind11#            calib.setExptime(exptime)
#pybind11#            calib.setFluxMag0(mag0, mag0Sigma)
#pybind11#
#pybind11#            calibs.append(calib)
#pybind11#
#pybind11#        ocalib = afwImage.Calib(calibs)
#pybind11#
#pybind11#        self.assertEqual(ocalib.getExptime(), ncalib*exptime)
#pybind11#        self.assertAlmostEqual(calibs[ncalib//2].getMidTime().get(), ocalib.getMidTime().get())
#pybind11#        # Check that we can only merge Calibs with the same fluxMag0 values
#pybind11#        calibs[0].setFluxMag0(1.001*mag0, mag0Sigma)
#pybind11#        with self.assertRaises(pexExcept.InvalidParameterError):
#pybind11#            afwImage.Calib(calibs)
#pybind11#
#pybind11#    def testCalibNegativeFlux(self):
#pybind11#        """Check that we can control if negative fluxes raise exceptions"""
#pybind11#        self.calib.setFluxMag0(1e12)
#pybind11#
#pybind11#        with self.assertRaises(pexExcept.DomainError):
#pybind11#            self.calib.getMagnitude(-10)
#pybind11#        with self.assertRaises(pexExcept.DomainError):
#pybind11#            self.calib.getMagnitude(-10, 1)
#pybind11#
#pybind11#        afwImage.Calib.setThrowOnNegativeFlux(False)
#pybind11#        mags = self.calib.getMagnitude(-10)
#pybind11#        self.assertTrue(np.isnan(mags))
#pybind11#        mags = self.calib.getMagnitude(-10, 1)
#pybind11#        self.assertTrue(np.isnan(mags[0]))
#pybind11#        self.assertTrue(np.isnan(mags[1]))
#pybind11#
#pybind11#        afwImage.Calib.setThrowOnNegativeFlux(True)
#pybind11#
#pybind11#        # Re-check that we raise after resetting ThrowOnNegativeFlux.
#pybind11#        with self.assertRaises(pexExcept.DomainError):
#pybind11#            self.calib.getMagnitude(-10)
#pybind11#        with self.assertRaises(pexExcept.DomainError):
#pybind11#            self.calib.getMagnitude(-10, 1)
#pybind11#
#pybind11#
#pybind11#def defineSdssFilters(testCase):
#pybind11#    """Initialise filters as used for our tests"""
#pybind11#    imageUtils.resetFilters()
#pybind11#    wavelengths = dict()
#pybind11#    testCase.aliases = dict(u=[], g=[], r=[], i=[], z=['zprime', "z'"])
#pybind11#    for name, lambdaEff in (('u', 355.1), ('g', 468.6), ('r', 616.5), ('i', 748.1), ('z', 893.1)):
#pybind11#        wavelengths[name] = lambdaEff
#pybind11#        imageUtils.defineFilter(name, lambdaEff, alias=testCase.aliases[name])
#pybind11#    return wavelengths
#pybind11#
#pybind11#
#pybind11#class ColorTestCase(lsst.utils.tests.TestCase):
#pybind11#    def setUp(self):
#pybind11#        defineSdssFilters(self)
#pybind11#
#pybind11#    def testCtor(self):
#pybind11#        afwImage.Color()
#pybind11#        afwImage.Color(1.2)
#pybind11#
#pybind11#    def testLambdaEff(self):
#pybind11#        f = afwImage.Filter("g")
#pybind11#        g_r = 1.2
#pybind11#        c = afwImage.Color(g_r)
#pybind11#
#pybind11#        self.assertEqual(c.getLambdaEff(f), 1000*g_r)  # XXX Not a real implementation!
#pybind11#
#pybind11#    def testIsIndeterminate(self):
#pybind11#        """Test that a default-constructed Color tests True, but ones with a g-r value test False"""
#pybind11#        self.assertTrue(afwImage.Color().isIndeterminate())
#pybind11#        self.assertFalse(afwImage.Color(1.2).isIndeterminate())
#pybind11#
#pybind11#
#pybind11#class FilterTestCase(lsst.utils.tests.TestCase):
#pybind11#    def setUp(self):
#pybind11#        # Initialise our filters
#pybind11#        # Start by forgetting that we may already have defined filters
#pybind11#        wavelengths = defineSdssFilters(self)
#pybind11#        self.filters = tuple(sorted(wavelengths.keys()))
#pybind11#        self.g_lambdaEff = [lambdaEff for name,
#pybind11#                            lambdaEff in wavelengths.items() if name == "g"][0]  # for tests
#pybind11#
#pybind11#    def defineFilterProperty(self, name, lambdaEff, force=False):
#pybind11#        return afwImage.FilterProperty(name, lambdaEff, force)
#pybind11#
#pybind11#    def testListFilters(self):
#pybind11#        self.assertEqual(afwImage.Filter_getNames(), self.filters)
#pybind11#
#pybind11#    def testCtorFromMetadata(self):
#pybind11#        """Test building a Filter from metadata"""
#pybind11#
#pybind11#        metadata = dafBase.PropertySet()
#pybind11#        metadata.add("FILTER", "g")
#pybind11#
#pybind11#        f = afwImage.Filter(metadata)
#pybind11#        self.assertEqual(f.getName(), "g")
#pybind11#        # Check that we can clean up metadata
#pybind11#        afwImage.stripFilterKeywords(metadata)
#pybind11#        self.assertEqual(len(metadata.names()), 0)
#pybind11#
#pybind11#        badFilter = "rhl"               # an undefined filter
#pybind11#        metadata.add("FILTER", badFilter)
#pybind11#        # Not defined
#pybind11#        with self.assertRaises(pexExcept.NotFoundError):
#pybind11#            afwImage.Filter(metadata)
#pybind11#
#pybind11#        # Force definition
#pybind11#        f = afwImage.Filter(metadata, True)
#pybind11#        self.assertEqual(f.getName(), badFilter)  # name is correctly defined
#pybind11#
#pybind11#    def testFilterEquality(self):
#pybind11#        """Test a "g" filter comparison"""
#pybind11#        f = afwImage.Filter("g")
#pybind11#        g = afwImage.Filter("g")
#pybind11#
#pybind11#        self.assertEqual(f, g)
#pybind11#
#pybind11#        f = afwImage.Filter()           # the unknown filter
#pybind11#        self.assertNotEqual(f, f)       # ... doesn't equal itself
#pybind11#
#pybind11#    def testFilterProperty(self):
#pybind11#        """Test properties of a "g" filter"""
#pybind11#        f = afwImage.Filter("g")
#pybind11#        # The properties of a g filter
#pybind11#        g = afwImage.FilterProperty.lookup("g")
#pybind11#
#pybind11#        self.assertEqual(f.getName(), "g")
#pybind11#        self.assertEqual(f.getId(), 1)
#pybind11#        self.assertEqual(f.getFilterProperty().getLambdaEff(), self.g_lambdaEff)
#pybind11#        self.assertEqual(f.getFilterProperty(), self.defineFilterProperty("gX", self.g_lambdaEff, True))
#pybind11#        self.assertEqual(g.getLambdaEff(), self.g_lambdaEff)
#pybind11#
#pybind11#    def testFilterAliases(self):
#pybind11#        """Test that we can provide an alias for a Filter"""
#pybind11#        for name0 in self.aliases:
#pybind11#            f0 = afwImage.Filter(name0)
#pybind11#            self.assertEqual(f0.getCanonicalName(), name0)
#pybind11#            self.assertEqual(sorted(f0.getAliases()), sorted(self.aliases[name0]))
#pybind11#
#pybind11#            for name in self.aliases[name0]:
#pybind11#                f = afwImage.Filter(name)
#pybind11#                self.assertEqual(sorted(f.getAliases()), sorted(self.aliases[name0]))
#pybind11#                self.assertEqual(f.getId(), f0.getId())
#pybind11#                self.assertEqual(f.getName(), name)
#pybind11#                self.assertEqual(afwImage.Filter(f.getId()).getName(), name0)
#pybind11#                self.assertEqual(f.getCanonicalName(), name0)
#pybind11#                self.assertNotEqual(f.getCanonicalName(), name)
#pybind11#                self.assertEqual(f.getFilterProperty().getLambdaEff(), f0.getFilterProperty().getLambdaEff())
#pybind11#
#pybind11#    def testReset(self):
#pybind11#        """Test that we can reset filter IDs and properties if needs be"""
#pybind11#        g = afwImage.FilterProperty.lookup("g")
#pybind11#
#pybind11#        # Can we add a filter property?
#pybind11#        with self.assertRaises(pexExcept.RuntimeError):
#pybind11#            self.defineFilterProperty("g", self.g_lambdaEff + 10)
#pybind11#        self.defineFilterProperty("g", self.g_lambdaEff + 10, True)  # should not raise
#pybind11#        self.defineFilterProperty("g", self.g_lambdaEff, True)
#pybind11#
#pybind11#        # Can we redefine properties?
#pybind11#        with self.assertRaises(pexExcept.RuntimeError):
#pybind11#            self.defineFilterProperty("g", self.g_lambdaEff + 10)  # changing definition is not allowed
#pybind11#
#pybind11#        self.defineFilterProperty("g", self.g_lambdaEff)  # identical redefinition is allowed
#pybind11#
#pybind11#        afwImage.Filter.define(g, afwImage.Filter("g").getId())  # OK if Id's the same
#pybind11#        afwImage.Filter.define(g, afwImage.Filter.AUTO)         # AUTO will assign the same ID
#pybind11#
#pybind11#        with self.assertRaises(pexExcept.RuntimeError):
#pybind11#            afwImage.Filter.define(g, afwImage.Filter("g").getId() + 10)  # different ID
#pybind11#
#pybind11#    def testUnknownFilter(self):
#pybind11#        """Test that we can define, but not use, an unknown filter"""
#pybind11#        badFilter = "rhl"               # an undefined filter
#pybind11#        with self.assertRaises(pexExcept.NotFoundError):
#pybind11#            afwImage.Filter(badFilter)
#pybind11#        # Force definition
#pybind11#        f = afwImage.Filter(badFilter, True)
#pybind11#        self.assertEqual(f.getName(), badFilter)  # name is correctly defined
#pybind11#
#pybind11#        # can't use Filter f
#pybind11#        with self.assertRaises(pexExcept.NotFoundError):
#pybind11#            f.getFilterProperty().getLambdaEff()
#pybind11#
#pybind11#        # Now define badFilter
#pybind11#        lambdaEff = 666.0
#pybind11#        self.defineFilterProperty(badFilter, lambdaEff)
#pybind11#
#pybind11#        self.assertEqual(f.getFilterProperty().getLambdaEff(), lambdaEff)  # but now we can
#pybind11#
#pybind11#        # Check that we didn't accidently define the unknown filter
#pybind11#        with self.assertRaises(pexExcept.NotFoundError):
#pybind11#            afwImage.Filter().getFilterProperty().getLambdaEff()
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
