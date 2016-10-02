#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
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
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.daf.base as dafBase
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.utils.tests
#pybind11#
#pybind11#
#pybind11#def headerToPropertyList(header):
#pybind11#    pl = dafBase.PropertyList()
#pybind11#    for key, value in header.items():
#pybind11#        pl.add(key, value)
#pybind11#    return pl
#pybind11#
#pybind11#
#pybind11#class WcsTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        # Actual WCS from processing Suprime-Cam
#pybind11#        self.width = 2048
#pybind11#        self.height = 4177
#pybind11#        self.header = {'NAXIS': 2,
#pybind11#                       'EQUINOX': 2000.0000000000,
#pybind11#                       'RADESYS': "FK5",
#pybind11#                       'CRPIX1': -3232.7544925483,
#pybind11#                       'CRPIX2': 4184.4881091129,
#pybind11#                       'CD1_1': -5.6123808607273e-05,
#pybind11#                       'CD1_2': 2.8951544956703e-07,
#pybind11#                       'CD2_1': 2.7343044348306e-07,
#pybind11#                       'CD2_2': 5.6100888336445e-05,
#pybind11#                       'CRVAL1': 5.6066137655191,
#pybind11#                       'CRVAL2': -0.60804032498548,
#pybind11#                       'CUNIT1': "deg",
#pybind11#                       'CUNIT2': "deg",
#pybind11#                       'A_ORDER': 5,
#pybind11#                       'A_0_2': 1.9749832126246e-08,
#pybind11#                       'A_0_3': 9.3734869173527e-12,
#pybind11#                       'A_0_4': 1.8812994578840e-17,
#pybind11#                       'A_0_5': -2.3524013652433e-19,
#pybind11#                       'A_1_1': -9.8443908806559e-10,
#pybind11#                       'A_1_2': -4.9278297504858e-10,
#pybind11#                       'A_1_3': -2.8491604610001e-16,
#pybind11#                       'A_1_4': 2.3185723720750e-18,
#pybind11#                       'A_2_0': 4.9546089730708e-08,
#pybind11#                       'A_2_1': -8.8592221672777e-12,
#pybind11#                       'A_2_2': 3.3560100338765e-16,
#pybind11#                       'A_2_3': 3.0469486185035e-21,
#pybind11#                       'A_3_0': -4.9332471706700e-10,
#pybind11#                       'A_3_1': -5.3126029725748e-16,
#pybind11#                       'A_3_2': 4.7795824885726e-18,
#pybind11#                       'A_4_0': 1.3128844828963e-16,
#pybind11#                       'A_4_1': 4.4014452170715e-19,
#pybind11#                       'A_5_0': 2.1781986904162e-18,
#pybind11#                       'B_ORDER': 5,
#pybind11#                       'B_0_2': -1.0607653075899e-08,
#pybind11#                       'B_0_3': -4.8693887937365e-10,
#pybind11#                       'B_0_4': -1.0363305097301e-15,
#pybind11#                       'B_0_5': 1.9621640066919e-18,
#pybind11#                       'B_1_1': 3.0340657679481e-08,
#pybind11#                       'B_1_2': -5.0763819284853e-12,
#pybind11#                       'B_1_3': 2.8987281654754e-16,
#pybind11#                       'B_1_4': 1.8253389678593e-19,
#pybind11#                       'B_2_0': -2.4772849184248e-08,
#pybind11#                       'B_2_1': -4.9775588352207e-10,
#pybind11#                       'B_2_2': -3.6806326254887e-16,
#pybind11#                       'B_2_3': 4.4136985315418e-18,
#pybind11#                       'B_3_0': -1.7807191001742e-11,
#pybind11#                       'B_3_1': -2.4136396882531e-16,
#pybind11#                       'B_3_2': 2.9165413645768e-19,
#pybind11#                       'B_4_0': 4.1029951148438e-16,
#pybind11#                       'B_4_1': 2.3711874424169e-18,
#pybind11#                       'B_5_0': 4.9333635889310e-19,
#pybind11#                       'AP_ORDER': 5,
#pybind11#                       'AP_0_1': -5.9740855298291e-06,
#pybind11#                       'AP_0_2': -2.0433429597268e-08,
#pybind11#                       'AP_0_3': -8.6810071023434e-12,
#pybind11#                       'AP_0_4': -2.4974690826778e-17,
#pybind11#                       'AP_0_5': 1.9819631102516e-19,
#pybind11#                       'AP_1_0': -4.5896648256716e-05,
#pybind11#                       'AP_1_1': -1.5248993348644e-09,
#pybind11#                       'AP_1_2': 5.0283116166943e-10,
#pybind11#                       'AP_1_3': 4.3796281513144e-16,
#pybind11#                       'AP_1_4': -2.1447889127908e-18,
#pybind11#                       'AP_2_0': -4.7550300344365e-08,
#pybind11#                       'AP_2_1': 1.0924172283232e-11,
#pybind11#                       'AP_2_2': -4.9862026098260e-16,
#pybind11#                       'AP_2_3': -5.4470851768869e-20,
#pybind11#                       'AP_3_0': 5.0130654116966e-10,
#pybind11#                       'AP_3_1': 6.8649554020012e-16,
#pybind11#                       'AP_3_2': -4.2759588436342e-18,
#pybind11#                       'AP_4_0': -3.6306802581471e-16,
#pybind11#                       'AP_4_1': -5.3885285875084e-19,
#pybind11#                       'AP_5_0': -1.8802693525108e-18,
#pybind11#                       'BP_ORDER': 5,
#pybind11#                       'BP_0_1': -2.6627855995942e-05,
#pybind11#                       'BP_0_2': 1.1143451873584e-08,
#pybind11#                       'BP_0_3': 4.9323396530135e-10,
#pybind11#                       'BP_0_4': 1.1785185735421e-15,
#pybind11#                       'BP_0_5': -1.6169957016415e-18,
#pybind11#                       'BP_1_0': -5.7914490267576e-06,
#pybind11#                       'BP_1_1': -3.0565765766244e-08,
#pybind11#                       'BP_1_2': 5.7727475030971e-12,
#pybind11#                       'BP_1_3': -4.0586821113726e-16,
#pybind11#                       'BP_1_4': -2.0662723654322e-19,
#pybind11#                       'BP_2_0': 2.3705520015164e-08,
#pybind11#                       'BP_2_1': 5.0530823594352e-10,
#pybind11#                       'BP_2_2': 3.8904979943489e-16,
#pybind11#                       'BP_2_3': -3.8346209540986e-18,
#pybind11#                       'BP_3_0': 1.9505421473262e-11,
#pybind11#                       'BP_3_1': 1.7583146713289e-16,
#pybind11#                       'BP_3_2': -3.4876779564534e-19,
#pybind11#                       'BP_4_0': -3.3690937119054e-16,
#pybind11#                       'BP_4_1': -2.0853007589561e-18,
#pybind11#                       'BP_5_0': -5.5344298912288e-19,
#pybind11#                       'CTYPE1': "RA---TAN-SIP",
#pybind11#                       'CTYPE2': "DEC--TAN-SIP",
#pybind11#                       }
#pybind11#
#pybind11#    def testInputInvariance(self):
#pybind11#        pl = headerToPropertyList(self.header)
#pybind11#        afwImage.makeWcs(pl)
#pybind11#        for key, value in self.header.items():
#pybind11#            self.assertEqual(value, pl.get(key), "%s not invariant: %s vs %s" % (key, value, pl.get(key)))
#pybind11#
#pybind11#    def testRepeat(self):
#pybind11#        pl = headerToPropertyList(self.header)
#pybind11#        wcs1 = afwImage.makeWcs(pl)
#pybind11#        wcs2 = afwImage.makeWcs(pl)
#pybind11#        for x, y in ((0, 0), (0, self.height), (self.width, 0), (self.width, self.height)):
#pybind11#            point = afwGeom.Point2D(x, y)
#pybind11#            self.assertEqual(wcs1.pixelToSky(point), wcs2.pixelToSky(point))
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
