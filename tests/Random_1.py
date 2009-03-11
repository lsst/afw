#!/usr/bin/env python
"""
Tests for the lsst.afw.math.Random Python wrapper

Run with:
   python Random_1.py
or
   python
   >>> import unittest; T=load("Random_1"); unittest.TextTestRunner(verbosity=1).run(T.suite())
"""
import os
import pdb
import time
import unittest

import lsst.pex.exceptions as pexExcept
import lsst.pex.policy as pexPolicy
import lsst.utils.tests as utilsTests
import lsst.afw.math as afwMath

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def checkRngEquivalence(rng1, rng2):
    for i in xrange(1000):
        assert rng1.get() == rng2.get()

def getSeed():
    return int(time.time() * 1000000.0) % 1000000

class RandomTestCase(unittest.TestCase):
    """A test case for lsst.afw.math.Random"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testCreate(self):
        rngs = []
        for name in afwMath.Random.getAlgorithmNames():
            rngs.append(afwMath.Random(name))
        for r in rngs:
            assert r.getMin() < r.getMax()
            assert afwMath.Random(r.getAlgorithmName()).get() == r.get()
            r2 = afwMath.Random(r.getAlgorithm())
            r2.get()
            assert r2.get() == r.get()

    def testCopy(self):
        """
        Test that the generator returned by deepCopy() produces an
        identical random number sequence to its prototype
        """
        rng1 = afwMath.Random(afwMath.Random.MT19937, getSeed())
        rng2 = rng1.deepCopy()
        checkRngEquivalence(rng1, rng2)

    def testOverride(self):
        """
        Tests that policy files and environment variables can override
        user specified algorithms and seed values.
        """
        pol = pexPolicy.Policy()
        emptyPol = pexPolicy.Policy()
        seed = getSeed()
        pol.set("rngSeed", str(seed))
        pol.set("rngAlgorithm", afwMath.Random.getAlgorithmNames()[afwMath.Random.RANLXD2])
        if (os.environ.has_key("LSST_RNG_ALGORITHM") and os.environ.has_key("LSST_RNG_SEED")):
            ref1 = afwMath.Random(os.environ["LSST_RNG_ALGORITHM"],
                                  long(os.environ["LSST_RNG_SEED"]))
        else:
            ref1 = afwMath.Random(afwMath.Random.MT19937, 53)
        ref2 = afwMath.Random(afwMath.Random.RANLXD2, seed)
        r1 = afwMath.Random.create(emptyPol, afwMath.Random.MT19937, 53)
        r2 = afwMath.Random.create(pol, afwMath.Random.MT19937, 53)
        checkRngEquivalence(ref1, r1)
        checkRngEquivalence(ref2, r2)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(RandomTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

if __name__ == "__main__":
    utilsTests.run(suite())

