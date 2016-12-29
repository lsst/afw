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

# -*- python -*-
"""
Check that coord and coordPtr are properly passed through swig

Run with:
   python coordptr.py
"""

from __future__ import absolute_import, division, print_function
import os
import unittest

import lsst.utils
import lsst.afw.image as image
import lsst.afw.geom as afwGeom
import lsst.afw.coord.coordLib as coord
import lsst.utils.tests


class CoordPtrTestCase(unittest.TestCase):

    def testMakeCoord(self):
        c = coord.Coord(1 * afwGeom.degrees, 2 * afwGeom.degrees)
        print(type(c))
        c = coord.makeCoord(coord.FK5, 1 * afwGeom.degrees, 2 * afwGeom.degrees)
        print(type(c))

    def testMakeWcs(self):
        afwdataDir = lsst.utils.getPackageDir("afw")
        path = os.path.join(afwdataDir, "tests", "data", "parent.fits")
        fitsHdr = image.readMetadata(path)

        wcs = image.makeWcs(fitsHdr)

        c = wcs.pixelToSky(0, 0)
        print(type(c))
        c.getPosition()

    def testCoordCast(self):
        for CoordClass in (coord.IcrsCoord, coord.Fk5Coord, coord.GalacticCoord, coord.EclipticCoord):
            derived1 = CoordClass(1 * afwGeom.degrees, 2 * afwGeom.degrees)
            self.assertEqual(type(derived1), CoordClass)
            base = derived1.clone()
            self.assertEqual(type(base), coord.Coord)
            derived2 = CoordClass.cast(base)
            self.assertEqual(type(derived2), CoordClass)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
