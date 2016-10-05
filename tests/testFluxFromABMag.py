#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2014 LSST Corporation.
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
#pybind11#"""
#pybind11#Tests for lsst.afw.table.FluxFromABMagTable, etc.
#pybind11#"""
#pybind11#import unittest
#pybind11#import math
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.image as afwImage
#pybind11#
#pybind11#
#pybind11#def refABMagFromFlux(flux):
#pybind11#    return -(5.0/2.0) * math.log10(flux/3631.0)
#pybind11#
#pybind11#
#pybind11#def refABMagErrFromFluxErr(fluxErr, flux):
#pybind11#    return math.fabs(fluxErr/(-0.4*flux*math.log(10)))
#pybind11#
#pybind11#
#pybind11#class FluxFromABMagTableTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def testBasics(self):
#pybind11#        for flux in (1, 210, 3210, 43210, 543210):
#pybind11#            abMag = afwImage.abMagFromFlux(flux)
#pybind11#            self.assertAlmostEqual(abMag, refABMagFromFlux(flux))
#pybind11#            fluxRoundTrip = afwImage.fluxFromABMag(abMag)
#pybind11#            self.assertAlmostEqual(flux, fluxRoundTrip)
#pybind11#
#pybind11#            for fluxErrFrac in (0.001, 0.01, 0.1):
#pybind11#                fluxErr = flux * fluxErrFrac
#pybind11#                abMagErr = afwImage.abMagErrFromFluxErr(fluxErr, flux)
#pybind11#                self.assertAlmostEqual(abMagErr, refABMagErrFromFluxErr(fluxErr, flux))
#pybind11#                fluxErrRoundTrip = afwImage.fluxErrFromABMagErr(abMagErr, abMag)
#pybind11#                self.assertAlmostEqual(fluxErr, fluxErrRoundTrip)
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
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
