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

import os
import unittest

import lsst.utils
import lsst.afw.geom as afwGeom
from lsst.afw.fits import readMetadata
import lsst.utils.tests


class WCSTestRaWrap(unittest.TestCase):
    '''A test set for the RA=0 wrap-around'''

    def setUp(self):
        mydir = lsst.utils.getPackageDir('afw')
        self.assertIsNotNone(mydir)
        self.datadir = os.path.join(mydir, 'tests')

    def test1(self):
        wcsfn = os.path.join(self.datadir, 'imsim-v85518312-fu-R43-S12.wcs2')
        hdr = readMetadata(wcsfn)
        wcs1 = afwGeom.makeSkyWcs(hdr)

        crval = wcs1.getSkyOrigin()
        cd = wcs1.getCdMatrix()
        print(cd)
        crval_p = afwGeom.Point2D(crval.getLongitude().asDegrees(),
                                  crval.getLatitude().asDegrees())
        origin = wcs1.getPixelOrigin()
        print(crval_p)
        print(origin)
        wcs2 = afwGeom.makeSkyWcs(crpix=origin, crval=crval, cdMatrix=cd)

        for wcs in [wcs1, wcs2]:
            print(wcs)
            print('x, y, RA, Dec, pixscale("/pix), pixscale2')
            for x, y in [(0, 0), (300, 0), (350, 0), (360, 0), (370, 0), (380, 0), (400, 0)]:
                pixPos = afwGeom.PointD(x, y)
                radec = wcs.pixelToSky(pixPos)
                ra = radec.getLongitude().asDegrees()
                dec = radec.getLatitude().asDegrees()
                pixscale = wcs.getPixelScale(pixPos).asArcseconds()
                print(x, y, ra, dec, pixscale)
                self.assertLess(abs(pixscale - 0.2), 1e-3)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
