# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest

import numpy as np

import lsst.utils.tests
import lsst.daf.base as dafBase
import lsst.pex.exceptions as pexExcept
import lsst.afw.image as afwImage
import lsst.afw.image.utils as imageUtils


def defineSdssFilters(testCase):
    """Initialise filters as used for our tests"""
    imageUtils.resetFilters()
    wavelengths = dict()
    testCase.aliases = dict(u=[], g=[], r=[], i=[], z=['zprime', "z'"])
    for name, lambdaEff in (('u', 355.1),
                            ('g', 468.6),
                            ('r', 616.5),
                            ('i', 748.1),
                            ('z', 893.1)):
        wavelengths[name] = lambdaEff
        imageUtils.defineFilter(name, lambdaEff, alias=testCase.aliases[name])
    return wavelengths


class ColorTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        defineSdssFilters(self)

    def testCtor(self):
        afwImage.Color()
        afwImage.Color(1.2)

    def testLambdaEff(self):
        f = afwImage.Filter("g")
        g_r = 1.2
        c = afwImage.Color(g_r)

        # XXX Not a real implementation!
        self.assertEqual(c.getLambdaEff(f), 1000*g_r)

    def testIsIndeterminate(self):
        """Test that a default-constructed Color tests True, but ones with a g-r value test False"""
        self.assertTrue(afwImage.Color().isIndeterminate())
        self.assertFalse(afwImage.Color(1.2).isIndeterminate())


class FilterTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        # Initialise our filters
        # Start by forgetting that we may already have defined filters
        wavelengths = defineSdssFilters(self)
        self.filters = tuple(sorted(wavelengths.keys()))
        self.g_lambdaEff = [lambdaEff for name,
                            lambdaEff in wavelengths.items() if name == "g"][0]  # for tests

    def defineFilterProperty(self, name, lambdaEff, force=False):
        return afwImage.FilterProperty(name, lambdaEff, force=force)

    def testListFilters(self):
        self.assertEqual(afwImage.Filter.getNames(), list(self.filters))

    def testCtorFromMetadata(self):
        """Test building a Filter from metadata"""

        metadata = dafBase.PropertySet()
        metadata.add("FILTER", "g")

        f = afwImage.Filter(metadata)
        self.assertEqual(f.getName(), "g")
        # Check that we can clean up metadata
        afwImage.stripFilterKeywords(metadata)
        self.assertEqual(len(metadata.names()), 0)

        badFilter = "rhl"               # an undefined filter
        metadata.add("FILTER", badFilter)
        # Not defined
        with self.assertRaises(pexExcept.NotFoundError):
            afwImage.Filter(metadata)

        # Force definition
        f = afwImage.Filter(metadata, True)
        self.assertEqual(f.getName(), badFilter)  # name is correctly defined

    def testFilterEquality(self):
        """Test a "g" filter comparison"""
        f = afwImage.Filter("g")
        g = afwImage.Filter("g")

        self.assertEqual(f, g)

        f = afwImage.Filter()           # the unknown filter
        self.assertNotEqual(f, f)       # ... doesn't equal itself

    def testFilterProperty(self):
        """Test properties of a "g" filter"""
        f = afwImage.Filter("g")
        # The properties of a g filter
        g = afwImage.FilterProperty.lookup("g")

        self.assertEqual(f.getName(), "g")
        self.assertEqual(f.getId(), 1)
        self.assertEqual(f.getFilterProperty().getLambdaEff(),
                         self.g_lambdaEff)
        self.assertEqual(f.getFilterProperty(),
                         self.defineFilterProperty("gX", self.g_lambdaEff, force=True))
        self.assertEqual(g.getLambdaEff(), self.g_lambdaEff)

    def testLambdaMinMax(self):
        """Test additional properties for minimum and maximum wavelength for a filter."""
        filt = afwImage.Filter("g")
        # LambdaMin and LambdaMax are undefined for the test SDSS filter, and should return nan
        self.assertTrue(np.isnan(filt.getFilterProperty().getLambdaMin()))
        self.assertTrue(np.isnan(filt.getFilterProperty().getLambdaMax()))
        lambdaEff = 476.31
        lambdaMin = 405
        lambdaMax = 552
        imageUtils.defineFilter("gNew", lambdaEff, lambdaMin=lambdaMin, lambdaMax=lambdaMax)
        filtNew = afwImage.Filter("gNew")
        self.assertEqual(lambdaMin, filtNew.getFilterProperty().getLambdaMin())
        self.assertEqual(lambdaMax, filtNew.getFilterProperty().getLambdaMax())

    def testFilterAliases(self):
        """Test that we can provide an alias for a Filter"""
        for name0 in self.aliases:
            f0 = afwImage.Filter(name0)
            self.assertEqual(f0.getCanonicalName(), name0)
            self.assertEqual(sorted(f0.getAliases()),
                             sorted(self.aliases[name0]))

            for name in self.aliases[name0]:
                f = afwImage.Filter(name)
                self.assertEqual(sorted(f.getAliases()),
                                 sorted(self.aliases[name0]))
                self.assertEqual(f.getId(), f0.getId())
                self.assertEqual(f.getName(), name)
                self.assertEqual(afwImage.Filter(f.getId()).getName(), name0)
                self.assertEqual(f.getCanonicalName(), name0)
                self.assertNotEqual(f.getCanonicalName(), name)
                self.assertEqual(f.getFilterProperty().getLambdaEff(),
                                 f0.getFilterProperty().getLambdaEff())

    def testReset(self):
        """Test that we can reset filter IDs and properties if needs be"""
        g = afwImage.FilterProperty.lookup("g")

        # Can we add a filter property?
        with self.assertRaises(pexExcept.RuntimeError):
            self.defineFilterProperty("g", self.g_lambdaEff + 10)
        # should not raise
        self.defineFilterProperty("g", self.g_lambdaEff + 10, True)
        self.defineFilterProperty("g", self.g_lambdaEff, True)

        # Can we redefine properties?
        with self.assertRaises(pexExcept.RuntimeError):
            # changing definition is not allowed
            self.defineFilterProperty("g", self.g_lambdaEff + 10)

        # identical redefinition is allowed
        self.defineFilterProperty("g", self.g_lambdaEff)

        # OK if Id's the same
        afwImage.Filter.define(g, afwImage.Filter("g").getId())
        # AUTO will assign the same ID
        afwImage.Filter.define(g, afwImage.Filter.AUTO)

        # different ID
        with self.assertRaises(pexExcept.RuntimeError):
            afwImage.Filter.define(g, afwImage.Filter("g").getId() + 10)

    def testUnknownFilter(self):
        """Test that we can define, but not use, an unknown filter"""
        badFilter = "rhl"               # an undefined filter
        with self.assertRaises(pexExcept.NotFoundError):
            afwImage.Filter(badFilter)
        # Force definition
        f = afwImage.Filter(badFilter, True)
        self.assertEqual(f.getName(), badFilter)  # name is correctly defined

        # can't use Filter f
        with self.assertRaises(pexExcept.NotFoundError):
            f.getFilterProperty().getLambdaEff()

        # Now define badFilter
        lambdaEff = 666.0
        self.defineFilterProperty(badFilter, lambdaEff)

        # but now we can
        self.assertEqual(f.getFilterProperty().getLambdaEff(), lambdaEff)

        # Check that we didn't accidently define the unknown filter
        with self.assertRaises(pexExcept.NotFoundError):
            afwImage.Filter().getFilterProperty().getLambdaEff()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
