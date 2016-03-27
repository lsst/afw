#!/usr/bin/env python2
from __future__ import absolute_import, division
#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#
"""
Tests for lsst.afw.table.FluxFromABMagTable, etc.
"""
import unittest
import math

import lsst.utils.tests
import lsst.afw.image as afwImage

def refABMagFromFlux(flux):
    return -(5.0/2.0) * math.log10(flux/3631.0)

def refABMagErrFromFluxErr(fluxErr, flux):
    return math.fabs(fluxErr/(-0.4*flux*math.log(10)))

class FluxFromABMagTableTestCase(unittest.TestCase):
    def testBasics(self):
        for flux in (1, 210, 3210, 43210, 543210):
            abMag = afwImage.abMagFromFlux(flux)
            self.assertAlmostEqual(abMag, refABMagFromFlux(flux))
            fluxRoundTrip = afwImage.fluxFromABMag(abMag)
            self.assertAlmostEqual(flux, fluxRoundTrip)

            for fluxErrFrac in (0.001, 0.01, 0.1):
                fluxErr = flux * fluxErrFrac
                abMagErr = afwImage.abMagErrFromFluxErr(fluxErr, flux)
                self.assertAlmostEqual(abMagErr, refABMagErrFromFluxErr(fluxErr, flux))
                fluxErrRoundTrip = afwImage.fluxErrFromABMagErr(abMagErr, abMag)
                self.assertAlmostEqual(fluxErr, fluxErrRoundTrip)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(FluxFromABMagTableTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
