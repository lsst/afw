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
#pybind11## -*- python -*-
#pybind11#"""
#pybind11#Check that coord and coordPtr are properly passed through swig
#pybind11#
#pybind11#Run with:
#pybind11#   python coordptr.py
#pybind11#"""
#pybind11#
#pybind11#import os
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils
#pybind11#import lsst.afw.image as image
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.coord.coordLib as coord
#pybind11#import lsst.utils.tests
#pybind11#
#pybind11#
#pybind11#class CoordPtrTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def testMakeCoord(self):
#pybind11#        c = coord.Coord(1 * afwGeom.degrees, 2 * afwGeom.degrees)
#pybind11#        print(type(c))
#pybind11#        c = coord.makeCoord(coord.FK5, 1 * afwGeom.degrees, 2 * afwGeom.degrees)
#pybind11#        print(type(c))
#pybind11#
#pybind11#    def testMakeWcs(self):
#pybind11#        afwdataDir = lsst.utils.getPackageDir("afw")
#pybind11#        path = os.path.join(afwdataDir, "tests", "data", "parent.fits")
#pybind11#        fitsHdr = image.readMetadata(path)
#pybind11#
#pybind11#        wcs = image.makeWcs(fitsHdr)
#pybind11#
#pybind11#        c = wcs.pixelToSky(0, 0)
#pybind11#        print(type(c))
#pybind11#        c.getPosition()
#pybind11#
#pybind11#    def testCoordCast(self):
#pybind11#        for CoordClass in (coord.IcrsCoord, coord.Fk5Coord, coord.GalacticCoord, coord.EclipticCoord):
#pybind11#            derived1 = CoordClass(1 * afwGeom.degrees, 2 * afwGeom.degrees)
#pybind11#            self.assertEqual(type(derived1), CoordClass)
#pybind11#            base = derived1.clone()
#pybind11#            self.assertEqual(type(base), coord.Coord)
#pybind11#            derived2 = CoordClass.cast(base)
#pybind11#            self.assertEqual(type(derived2), CoordClass)
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
