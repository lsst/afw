#!/usr/bin/env python
"""
Tests for Color and Filter

Run with:
   color.py
or
   python
   >>> import color; color.run()
"""


import os, sys
import eups
import unittest
import lsst.utils.tests as tests
import lsst.pex.logging as logging
import lsst.pex.exceptions as pexExcept
import lsst.pex.policy as pexPolicy
import lsst.afw.image as afwImage
import lsst.afw.image.utils as imageUtils
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDetect
import lsst.afw.detection.utils as afwDetectUtils
import lsst.afw.display.ds9 as ds9

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

class ColorTestCase(unittest.TestCase):
    """A test case for Color"""
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testCtor(self):
        c = afwImage.Color()
        c = afwImage.Color(1.2)

    def testLambdaEff(self):
        f = afwImage.Filter(afwImage.Filter.G)
        g_r = 1.2
        c = afwImage.Color(g_r)

        self.assertEqual(c.getLambdaEff(f), 1000*g_r) # XXX Not a real implementation!

    def testBool(self):
        """Test that a default-constructed Color tests False, but ones with a g-r value test True"""
        self.assertFalse(afwImage.Color())
        self.assertTrue(afwImage.Color(1.2))

class FilterTestCase(unittest.TestCase):
    """A test case for Filter"""

    def setUp(self):
        # Initialise our filters
        #
        # Start by forgetting that we may already have defined filters
        #
        filterPolicy = pexPolicy.Policy.createPolicy(
            os.path.join(eups.productDir("afw"), "tests", "SdssFilters.paf"), True)

        imageUtils.defineFiltersFromPolicy(filterPolicy, reset=True)

        self.g_lambdaEff = [p.get("lambdaEff") for p in filterPolicy.getArray("Filter")
                            if p.get("name") == "g"][0] # used for tests

    def defineFilterProperty(self, name, lambdaEff, force=False):
        filterPolicy = pexPolicy.Policy()
        filterPolicy.add("lambdaEff", lambdaEff)

        return afwImage.FilterProperty(name, filterPolicy, force);

    def testCtor(self):
        """Test that we can construct a Filter"""
        # A filter of type 
        f = afwImage.Filter("g")

    def testFilterProperty(self):
        # a "g" filter
        f = afwImage.Filter("g")
        # The properties of a g filter
        g = afwImage.FilterProperty.lookup("g")

        if False:
            print "Filter: %s == %d lambda_{eff}=%g" % (f.getName(), f.getId(),
                                                        f.getFilterProperty().getLambdaEff())

        self.assertEqual(f.getName(), "g")
        self.assertEqual(f.getId(), 1)
        self.assertEqual(f.getFilterProperty().getLambdaEff(), self.g_lambdaEff)

        self.assertEqual(g.getLambdaEff(), self.g_lambdaEff)

    def testFilterAliases(self):
        """Test that we can provide an alias for a Filter"""
        f0 = afwImage.Filter("z")
        f1 = afwImage.Filter("zprime")
        f2 = afwImage.Filter("z'")

        self.assertEqual(f0.getFilterProperty().getLambdaEff(), f1.getFilterProperty().getLambdaEff())
        self.assertEqual(f0.getFilterProperty().getLambdaEff(), f2.getFilterProperty().getLambdaEff())

    def testReset(self):
        """Test that we can reset filter IDs and properties if needs be"""
        # The properties of a g filter
        g = afwImage.FilterProperty.lookup("g")
        #
        # First FilterProperty
        #
        def tst():
            gprime = self.defineFilterProperty("g", self.g_lambdaEff + 10)

        tests.assertRaisesLsstCpp(self, pexExcept.RuntimeErrorException, tst)
        gprime = self.defineFilterProperty("g", self.g_lambdaEff + 10, True) # should not raise
        gprime = self.defineFilterProperty("g", self.g_lambdaEff, True)
        #
        # Now Filter
        #
        def tst():
            afwImage.Filter.define(g, afwImage.Filter.AUTO)

        afwImage.Filter.define(g, afwImage.Filter("g").getId()) # OK if Id's the same
        tests.assertRaisesLsstCpp(self, pexExcept.RuntimeErrorException, tst)
        afwImage.Filter.define(g, afwImage.Filter.AUTO, True)

    def testUnknownFilter(self):
        """Test that we can define, but not use, an unknown filter"""
        badFilter = "rhl"               # an undefined filter
        # Not defined
        tests.assertRaisesLsstCpp(self, pexExcept.NotFoundException,
                                  lambda : afwImage.Filter(badFilter))
        # Force definition
        f = afwImage.Filter(badFilter, True)
        self.assertEqual(f.getName(), badFilter) # name is correctly defined
        
        tests.assertRaisesLsstCpp(self, pexExcept.NotFoundException,
                                  lambda : f.getFilterProperty().getLambdaEff()) # can't use Filter f
        #
        # Now define badFilter
        #
        lambdaEff = 666.0; self.defineFilterProperty(badFilter, lambdaEff)
        
        self.assertEqual(f.getFilterProperty().getLambdaEff(), lambdaEff) # but now we can
        #
        # Check that we didn't accidently define the unknown filter
        #
        tests.assertRaisesLsstCpp(self, pexExcept.NotFoundException,
                                  lambda : afwImage.Filter().getFilterProperty().getLambdaEff())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    if False:
        suites += unittest.makeSuite(ColorTestCase)
    else:
        print >> sys.stderr, "Skipping Color tests (wait until #1196 is merged)"
    suites += unittest.makeSuite(FilterTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
