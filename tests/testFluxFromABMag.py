#!/usr/bin/env python2
from __future__ import absolute_import, division
#
# LSST Data Management System
# Copyright 2014 LSST Corporation.
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
