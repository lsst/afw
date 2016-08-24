#!/usr/bin/env python
from __future__ import absolute_import, division

#
# LSST Data Management System
# Copyright 2016 LSST Corporation.
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

import numpy
import astropy.io.fits
import lsst.utils.tests
import lsst.afw.geom
import lsst.afw.table


class TableIoTestCase(lsst.utils.tests.TestCase):

    def testAngleUnitWriting(self):
        """Test that Angle columns have TUNIT set appropriately,
        as per DM-7221.
        """
        schema = lsst.afw.table.Schema()
        key = schema.addField("a", type="Angle", doc="angle field")
        outCat = lsst.afw.table.BaseCatalog(schema)
        outCat.addNew().set(key, 1.0*lsst.afw.geom.degrees)
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            outCat.writeFits(tmpFile)
            inFits = astropy.io.fits.open(tmpFile)
            self.assertEqual(inFits[1].header["TTYPE1"], "a")
            self.assertEqual(inFits[1].header["TUNIT1"], "rad")


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
