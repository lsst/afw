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
# This ticket results from
#   https://dev.lsstcorp.org/trac/ticket/2352
# "Support multi-extension input FITS files"

from __future__ import absolute_import, division, print_function
import os
import os.path
import unittest

import lsst.afw.image as afwImage
import lsst.utils.tests

testPath = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(testPath, "data", "ticket2352.fits")


class ReadMefTest(unittest.TestCase):
    """Test the reading of a multi-extension FITS (MEF) file"""

    def checkExtName(self, name, value, extNum):
        filename = DATA + "[%s]" % name

        header = afwImage.readMetadata(filename)
        self.assertEqual(header.get("EXT_NUM"), extNum)
        self.assertEqual(header.get("EXTNAME").strip(), name)

        image = afwImage.ImageI(filename)
        self.assertEqual(image.get(0, 0), value)

    def testExtName(self):
        self.checkExtName("ONE", 1, 2)
        self.checkExtName("TWO", 2, 3)
        self.checkExtName("THREE", 3, 4)

    def checkExtNum(self, hdu, extNum):
        if hdu is None:
            hdu = -(1 << 31)            # == INT_MIN
        header = afwImage.readMetadata(DATA, hdu)
        self.assertEqual(header.get("EXT_NUM"), extNum)

    def testExtNum(self):
        # N.b.  The test file was written with 1-indexed EXT_NUMs
        self.checkExtNum(None, 2)  # Should skip PHU
        self.checkExtNum(0, 1)
        self.checkExtNum(1, 2)
        self.checkExtNum(2, 3)
        self.checkExtNum(3, 4)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
