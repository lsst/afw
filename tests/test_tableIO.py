#
# LSST Data Management System
# Copyright 2016 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Soeftware Foundation, either version 3 of the License, or
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

import numpy as np
import astropy.io.fits

import lsst.utils.tests
import lsst.geom
import lsst.afw.table
import lsst.afw.image


class TableIoTestCase(lsst.utils.tests.TestCase):

    def testAngleUnitWriting(self):
        """Test that Angle columns have TUNIT set appropriately,
        as per DM-7221.
        """
        schema = lsst.afw.table.Schema()
        key = schema.addField("a", type="Angle", doc="angle field")
        outCat = lsst.afw.table.BaseCatalog(schema)
        outCat.addNew().set(key, 1.0*lsst.geom.degrees)
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            outCat.writeFits(tmpFile)
            inFits = astropy.io.fits.open(tmpFile)
            self.assertEqual(inFits[1].header["TTYPE1"], "a")
            self.assertEqual(inFits[1].header["TUNIT1"], "rad")
            inFits.close()

    def testSchemaReading(self):
        """Test that a Schema can be read from a FITS file

        Per DM-8211.
        """
        schema = lsst.afw.table.Schema()
        aa = schema.addField("a", type=np.int64, doc="a")
        bb = schema.addField("b", type=np.float64, doc="b")
        schema.getAliasMap().set("c", "a")
        schema.getAliasMap().set("d", "b")
        cat = lsst.afw.table.BaseCatalog(schema)
        row = cat.addNew()
        row.set(aa, 12345)
        row.set(bb, 1.2345)
        with lsst.utils.tests.getTempFilePath(".fits") as temp:
            cat.writeFits(temp)
            self.assertEqual(lsst.afw.table.Schema.readFits(temp), schema)
            # Not testing Schema.fromFitsMetadata because afw.fits.readMetadata (which is the only
            # python-accessible FITS header reader) returns a PropertySet, but we want a PropertyList
            # and it doesn't up-convert easily.

    def testPreppedRows(self):
        schema = lsst.afw.table.Schema()
        aa = schema.addField("a", type=np.float64, doc="a")
        bb = schema.addField("b", type=np.float64, doc="b")
        cc = schema.addField("c", type="ArrayF", doc="c", size=4)
        self.assertEqual(schema.getRecordSize(), 32)
        lsst.afw.table.io.setPreppedRowsFactor(128)
        # nRowsToPrep is PreppedRowsFactor/RecordSize = 4
        # Test that we can read catalogs both larger and smaller than this.
        nRowsToPrep = lsst.afw.table.io.getPreppedRowsFactor() // schema.getRecordSize()
        smaller = lsst.afw.table.BaseCatalog(schema)
        smaller.resize(nRowsToPrep - 1)
        smaller[aa] = np.random.randn(nRowsToPrep - 1)
        smaller[bb] = np.random.randn(nRowsToPrep - 1)
        smaller[cc] = np.random.randn(nRowsToPrep - 1, 4)
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            smaller.writeFits(tmpFile)
            smaller2 = lsst.afw.table.BaseCatalog.readFits(tmpFile)
            self.assertFloatsEqual(smaller[aa], smaller2[aa])
            self.assertFloatsEqual(smaller[bb], smaller2[bb])
            self.assertFloatsEqual(smaller[cc], smaller2[cc])
        larger = lsst.afw.table.BaseCatalog(schema)
        larger.resize(nRowsToPrep + 1)
        larger[aa] = np.random.randn(nRowsToPrep + 1)
        larger[bb] = np.random.randn(nRowsToPrep + 1)
        larger[cc] = np.random.randn(nRowsToPrep + 1, 4)
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            larger.writeFits(tmpFile)
            larger2 = lsst.afw.table.BaseCatalog.readFits(tmpFile)
            self.assertFloatsEqual(larger[aa], larger2[aa])
            self.assertFloatsEqual(larger[bb], larger2[bb])
            self.assertFloatsEqual(larger[cc], larger2[cc])


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
