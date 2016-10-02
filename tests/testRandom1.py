#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import str
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
#pybind11#"""
#pybind11#Tests for the lsst.afw.math.Random Python wrapper
#pybind11#
#pybind11#Run with:
#pybind11#   python Random_1.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import unittest; T=load("Random_1"); unittest.TextTestRunner(verbosity=1).run(T.suite())
#pybind11#"""
#pybind11#
#pybind11#import sys
#pybind11#import time
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.pex.policy as pexPolicy
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#def checkRngEquivalence(rng1, rng2):
#pybind11#    for i in range(1000):
#pybind11#        assert rng1.uniform() == rng2.uniform()
#pybind11#
#pybind11#
#pybind11#def getSeed():
#pybind11#    return int(time.time() * 1000000.0) % 1000000
#pybind11#
#pybind11#
#pybind11#class RandomTestCase(unittest.TestCase):
#pybind11#    """A test case for lsst.afw.math.Random"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        pass
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        pass
#pybind11#
#pybind11#    def testCreate(self):
#pybind11#        rngs = []
#pybind11#        for name in afwMath.Random.getAlgorithmNames():
#pybind11#            rngs.append(afwMath.Random(name))
#pybind11#        for r in rngs:
#pybind11#            assert afwMath.Random(r.getAlgorithmName()).uniform() == r.uniform()
#pybind11#            r2 = afwMath.Random(r.getAlgorithm())
#pybind11#            r2.uniform()
#pybind11#            assert r2.uniform() == r.uniform()
#pybind11#
#pybind11#    def testCopy(self):
#pybind11#        """
#pybind11#        Test that the generator returned by deepCopy() produces an
#pybind11#        identical random number sequence to its prototype
#pybind11#        """
#pybind11#        rng1 = afwMath.Random(afwMath.Random.MT19937, getSeed())
#pybind11#        rng2 = rng1.deepCopy()
#pybind11#        checkRngEquivalence(rng1, rng2)
#pybind11#
#pybind11#    def testPolicy(self):
#pybind11#        """
#pybind11#        Tests that policy files and environment variables can override
#pybind11#        user specified algorithms and seed values.
#pybind11#        """
#pybind11#        pol = pexPolicy.Policy()
#pybind11#        seed = getSeed()
#pybind11#        pol.set("rngSeed", str(seed))
#pybind11#        pol.set("rngAlgorithm", afwMath.Random.getAlgorithmNames()[afwMath.Random.RANLXD2])
#pybind11#        r1 = afwMath.Random(afwMath.Random.RANLXD2, seed)
#pybind11#        r2 = afwMath.Random(pol)
#pybind11#        checkRngEquivalence(r1, r2)
#pybind11#
#pybind11#
#pybind11#class RandomImageTestCase(unittest.TestCase):
#pybind11#    """A test case for lsst.afw.math.Random applied to Images"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.rand = afwMath.Random()
#pybind11#        self.image = afwImage.ImageF(afwGeom.Extent2I(1000, 1000))
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.image
#pybind11#
#pybind11#    def testState(self):
#pybind11#        for i in range(100):
#pybind11#            self.rand.uniformInt(10000)
#pybind11#        state = self.rand.getState()
#pybind11#        self.assertIsInstance(state, bytes)
#pybind11#        v1 = [self.rand.uniformInt(10000) for i in range(100)]
#pybind11#        self.rand.setState(state)
#pybind11#        v2 = [self.rand.uniformInt(10000) for i in range(100)]
#pybind11#        self.assertEqual(v1, v2)
#pybind11#
#pybind11#    def testStateRaisesType(self):
#pybind11#        with self.assertRaises(TypeError):
#pybind11#            self.rand.setState(self.rand)
#pybind11#
#pybind11#    @unittest.skipIf(sys.version_info.major < 3, "setState can not distinguish unicode from bytes")
#pybind11#    def testStateRaisesUnicodeType(self):
#pybind11#        with self.assertRaises(TypeError):
#pybind11#            self.rand.setState(u"\u03bc not bytes")
#pybind11#
#pybind11#    def testStateRaisesLength(self):
#pybind11#        with self.assertRaises(lsst.pex.exceptions.LengthError):
#pybind11#            self.rand.setState(b"too small")
#pybind11#
#pybind11#    def testRandomUniformImage(self):
#pybind11#        afwMath.randomUniformImage(self.image, self.rand)
#pybind11#        #stats = afwMath.makeStatistics(self.image, afwMath.MEAN | afwMath.STDEV)
#pybind11#
#pybind11#    def testRandomGaussianImage(self):
#pybind11#        afwMath.randomGaussianImage(self.image, self.rand)
#pybind11#        #stats = afwMath.makeStatistics(self.image, afwMath.MEAN | afwMath.STDEV)
#pybind11#
#pybind11#    def testRandomChisqImage(self):
#pybind11#        nu = 10
#pybind11#        afwMath.randomChisqImage(self.image, self.rand, nu)
#pybind11#        stats = afwMath.makeStatistics(self.image, afwMath.MEAN | afwMath.VARIANCE)
#pybind11#        if False:
#pybind11#            print("nu = %g.  mean = %g, variance = %g" %
#pybind11#                  (nu, stats.getValue(afwMath.MEAN), stats.getValue(afwMath.VARIANCE)))
#pybind11#        self.assertAlmostEqual(stats.getValue(afwMath.MEAN), nu, 1)
#pybind11#        self.assertAlmostEqual(stats.getValue(afwMath.VARIANCE), 2*nu, 1)
#pybind11#
#pybind11#    def testRandomPoissonImage(self):
#pybind11#        mu = 10
#pybind11#        afwMath.randomPoissonImage(self.image, self.rand, mu)
#pybind11#        stats = afwMath.makeStatistics(self.image, afwMath.MEAN | afwMath.VARIANCE)
#pybind11#        if False:
#pybind11#            print("mu = %g.  mean = %g, variance = %g" %
#pybind11#                  (mu, stats.getValue(afwMath.MEAN), stats.getValue(afwMath.VARIANCE)))
#pybind11#        self.assertAlmostEqual(stats.getValue(afwMath.MEAN), mu, 1)
#pybind11#        self.assertAlmostEqual(stats.getValue(afwMath.VARIANCE), mu, 1)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
