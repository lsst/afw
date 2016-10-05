#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
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
#pybind11#import os
#pybind11#import unittest
#pybind11#import lsst.utils
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.utils.tests
#pybind11#from math import sqrt
#pybind11#
#pybind11#
#pybind11#class WCSTestRaWrap(unittest.TestCase):
#pybind11#    '''A test set for the RA=0 wrap-around'''
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        mydir = lsst.utils.getPackageDir('afw')
#pybind11#        self.assertIsNotNone(mydir)
#pybind11#        self.datadir = os.path.join(mydir, 'tests')
#pybind11#
#pybind11#    def test1(self):
#pybind11#        # This fails due to #1386
#pybind11#        #wcsfn = os.path.join(self.datadir, 'imsim-v85518312-fu-R43-S12.wcs')
#pybind11#        wcsfn = os.path.join(self.datadir, 'imsim-v85518312-fu-R43-S12.wcs2')
#pybind11#        hdr = afwImage.readMetadata(wcsfn)
#pybind11#        wcs1 = afwImage.makeWcs(hdr)
#pybind11#
#pybind11#        crval = wcs1.getSkyOrigin()
#pybind11#        cd = wcs1.getCDMatrix()
#pybind11#        print(cd)
#pybind11#        crval_p = afwGeom.Point2D(crval.getLongitude().asDegrees(),
#pybind11#                                  crval.getLatitude().asDegrees())
#pybind11#        origin = wcs1.getPixelOrigin()
#pybind11#        print(crval_p)
#pybind11#        print(origin)
#pybind11#        wcs2 = afwImage.Wcs(crval_p, origin, cd)
#pybind11#
#pybind11#        for wcs in [wcs1, wcs2]:
#pybind11#            print(wcs)
#pybind11#            print('x, y, RA, Dec, pixscale("/pix), pixscale2')
#pybind11#            for x, y in [(0, 0), (300, 0), (350, 0), (360, 0), (370, 0), (380, 0), (400, 0)]:
#pybind11#                radec = wcs.pixelToSky(x, y)
#pybind11#                ra = radec.getLongitude().asDegrees()
#pybind11#                dec = radec.getLatitude().asDegrees()
#pybind11#                pixscale = 3600. * sqrt(wcs.pixArea(afwGeom.Point2D(x, y)))
#pybind11#                ps2 = wcs.pixelScale().asArcseconds()
#pybind11#                print(x, y, ra, dec, pixscale, ps2)
#pybind11#                self.assertLess(abs(pixscale - 0.2), 1e-3)
#pybind11#                self.assertLess(abs(ps2 - 0.2), 1e-3)
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
