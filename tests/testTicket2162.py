#!/usr/bin/env python
from __future__ import absolute_import, division

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

import unittest

import lsst.daf.base as dafBase
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.utils.tests as utilsTests


def headerToPropertyList(header):
    pl = dafBase.PropertyList()
    for key, value in header.items():
        pl.add(key, value)
    return pl


class WcsTestCase(unittest.TestCase):

    def setUp(self):
        # Actual WCS from processing Suprime-Cam
        self.width = 2048
        self.height = 4177
        self.header = {'NAXIS': 2,
                       'EQUINOX': 2000.0000000000,
                       'RADESYS': "FK5",
                       'CRPIX1': -3232.7544925483,
                       'CRPIX2': 4184.4881091129,
                       'CD1_1': -5.6123808607273e-05,
                       'CD1_2': 2.8951544956703e-07,
                       'CD2_1': 2.7343044348306e-07,
                       'CD2_2': 5.6100888336445e-05,
                       'CRVAL1': 5.6066137655191,
                       'CRVAL2': -0.60804032498548,
                       'CUNIT1': "deg",
                       'CUNIT2': "deg",
                       'A_ORDER': 5,
                       'A_0_2': 1.9749832126246e-08,
                       'A_0_3': 9.3734869173527e-12,
                       'A_0_4': 1.8812994578840e-17,
                       'A_0_5': -2.3524013652433e-19,
                       'A_1_1': -9.8443908806559e-10,
                       'A_1_2': -4.9278297504858e-10,
                       'A_1_3': -2.8491604610001e-16,
                       'A_1_4': 2.3185723720750e-18,
                       'A_2_0': 4.9546089730708e-08,
                       'A_2_1': -8.8592221672777e-12,
                       'A_2_2': 3.3560100338765e-16,
                       'A_2_3': 3.0469486185035e-21,
                       'A_3_0': -4.9332471706700e-10,
                       'A_3_1': -5.3126029725748e-16,
                       'A_3_2': 4.7795824885726e-18,
                       'A_4_0': 1.3128844828963e-16,
                       'A_4_1': 4.4014452170715e-19,
                       'A_5_0': 2.1781986904162e-18,
                       'B_ORDER': 5,
                       'B_0_2': -1.0607653075899e-08,
                       'B_0_3': -4.8693887937365e-10,
                       'B_0_4': -1.0363305097301e-15,
                       'B_0_5': 1.9621640066919e-18,
                       'B_1_1': 3.0340657679481e-08,
                       'B_1_2': -5.0763819284853e-12,
                       'B_1_3': 2.8987281654754e-16,
                       'B_1_4': 1.8253389678593e-19,
                       'B_2_0': -2.4772849184248e-08,
                       'B_2_1': -4.9775588352207e-10,
                       'B_2_2': -3.6806326254887e-16,
                       'B_2_3': 4.4136985315418e-18,
                       'B_3_0': -1.7807191001742e-11,
                       'B_3_1': -2.4136396882531e-16,
                       'B_3_2': 2.9165413645768e-19,
                       'B_4_0': 4.1029951148438e-16,
                       'B_4_1': 2.3711874424169e-18,
                       'B_5_0': 4.9333635889310e-19,
                       'AP_ORDER': 5,
                       'AP_0_1': -5.9740855298291e-06,
                       'AP_0_2': -2.0433429597268e-08,
                       'AP_0_3': -8.6810071023434e-12,
                       'AP_0_4': -2.4974690826778e-17,
                       'AP_0_5': 1.9819631102516e-19,
                       'AP_1_0': -4.5896648256716e-05,
                       'AP_1_1': -1.5248993348644e-09,
                       'AP_1_2': 5.0283116166943e-10,
                       'AP_1_3': 4.3796281513144e-16,
                       'AP_1_4': -2.1447889127908e-18,
                       'AP_2_0': -4.7550300344365e-08,
                       'AP_2_1': 1.0924172283232e-11,
                       'AP_2_2': -4.9862026098260e-16,
                       'AP_2_3': -5.4470851768869e-20,
                       'AP_3_0': 5.0130654116966e-10,
                       'AP_3_1': 6.8649554020012e-16,
                       'AP_3_2': -4.2759588436342e-18,
                       'AP_4_0': -3.6306802581471e-16,
                       'AP_4_1': -5.3885285875084e-19,
                       'AP_5_0': -1.8802693525108e-18,
                       'BP_ORDER': 5,
                       'BP_0_1': -2.6627855995942e-05,
                       'BP_0_2': 1.1143451873584e-08,
                       'BP_0_3': 4.9323396530135e-10,
                       'BP_0_4': 1.1785185735421e-15,
                       'BP_0_5': -1.6169957016415e-18,
                       'BP_1_0': -5.7914490267576e-06,
                       'BP_1_1': -3.0565765766244e-08,
                       'BP_1_2': 5.7727475030971e-12,
                       'BP_1_3': -4.0586821113726e-16,
                       'BP_1_4': -2.0662723654322e-19,
                       'BP_2_0': 2.3705520015164e-08,
                       'BP_2_1': 5.0530823594352e-10,
                       'BP_2_2': 3.8904979943489e-16,
                       'BP_2_3': -3.8346209540986e-18,
                       'BP_3_0': 1.9505421473262e-11,
                       'BP_3_1': 1.7583146713289e-16,
                       'BP_3_2': -3.4876779564534e-19,
                       'BP_4_0': -3.3690937119054e-16,
                       'BP_4_1': -2.0853007589561e-18,
                       'BP_5_0': -5.5344298912288e-19,
                       'CTYPE1': "RA---TAN-SIP",
                       'CTYPE2': "DEC--TAN-SIP",
                       }

    def testInputInvariance(self):
        pl = headerToPropertyList(self.header)
        afwImage.makeWcs(pl)
        for key, value in self.header.items():
            self.assertEqual(value, pl.get(key), "%s not invariant: %s vs %s" % (key, value, pl.get(key)))

    def testRepeat(self):
        pl = headerToPropertyList(self.header)
        wcs1 = afwImage.makeWcs(pl)
        wcs2 = afwImage.makeWcs(pl)
        for x, y in ((0, 0), (0, self.height), (self.width, 0), (self.width, self.height)):
            point = afwGeom.Point2D(x, y)
            self.assertEqual(wcs1.pixelToSky(point), wcs2.pixelToSky(point))


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(WcsTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
