#!/usr/bin/env python
from __future__ import absolute_import, division

#
# LSST Data Management System
# Copyright 2016 AURA/LSST
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

"""
Tests for Astropy views into afw.table Catalogs

Run with:
   ./testAstropyTableViews.py
or
   python
   >>> import testAstropyTableViews; testAstropyTableViews.run()
"""

import unittest
import operator

import numpy
import astropy.table
import astropy.units

import lsst.utils.tests
import lsst.afw.table


class AstropyTableViewTestCase(lsst.utils.tests.TestCase):
    """Test that we can construct Astropy views to afw.table Catalog objects.

    This test case does not yet test the syntax

        table = astropy.table.Table(lsst_catalog)

    which is made available by BaseCatalog.__astropy_table__, as this will not
    be available until Astropy 1.2 is released.  However, this simply
    delegates to BaseCatalog.asAstropy, which can also be called directly.
    """

    def setUp(self):
        schema = lsst.afw.table.Schema()
        self.k1 = schema.addField("a1", type=float, units="meter", doc="a1 (meter)")
        self.k2 = schema.addField("a2", type=int, doc="a2 (unitless)")
        self.k3 = schema.addField("a3", type="ArrayF", size=3, units="count", doc="a3 (array, counts)")
        self.k4 = schema.addField("a4", type="Flag", doc="a4 (flag)")
        self.k5 = lsst.afw.table.CoordKey.addFields(schema, "a5", "a5 coordinate")
        self.k6 = schema.addField("a6", type=str, size=8, doc="a6 (str)")
        self.catalog = lsst.afw.table.BaseCatalog(schema)
        self.data = [
            {
                "a1": 5.0, "a2": 3, "a3": numpy.array([0.5, 0.0, -0.5], dtype=numpy.float32),
                "a4": True, "a5_ra": 45.0*lsst.afw.geom.degrees, "a5_dec": 30.0*lsst.afw.geom.degrees,
                "a6": "bubbles"
            },
            {
                "a1": 2.5, "a2": 7, "a3": numpy.array([1.0, 0.5, -1.5], dtype=numpy.float32),
                "a4": False, "a5_ra": 25.0*lsst.afw.geom.degrees, "a5_dec": -60.0*lsst.afw.geom.degrees,
                "a6": "pingpong"
            },
        ]
        for d in self.data:
            record = self.catalog.addNew()
            for k, v in d.items():
                record.set(k, v)

    def tearDown(self):
        del self.k1
        del self.k2
        del self.k3
        del self.k4
        del self.k5
        del self.k6
        del self.catalog
        del self.data

    def testQuantityColumn(self):
        """Test that a column with units is handled as expected by Table and QTable.
        """
        v1 = self.catalog.asAstropy(cls=astropy.table.Table, unviewable="skip")
        self.assertEqual(v1["a1"].unit, astropy.units.Unit("m"))
        self.assertClose(v1["a1"], self.catalog["a1"])
        self.assertNotIsInstance(v1["a1"], astropy.units.Quantity)
        v2 = self.catalog.asAstropy(cls=astropy.table.QTable, unviewable="skip")
        self.assertEqual(v2["a1"].unit, astropy.units.Unit("m"))
        self.assertClose(v2["a1"]/astropy.units.Quantity(self.catalog["a1"]*100, "cm"), 1.0)
        self.assertIsInstance(v2["a1"], astropy.units.Quantity)

    def testUnitlessColumn(self):
        """Test that a column without units is handled as expected by Table and QTable.
        """
        v1 = self.catalog.asAstropy(cls=astropy.table.Table, unviewable="skip")
        self.assertEqual(v1["a2"].unit, None)
        self.assertClose(v1["a2"], self.catalog["a2"])  # use assertClose just because it handles arrays
        v2 = self.catalog.asAstropy(cls=astropy.table.QTable, unviewable="skip")
        self.assertEqual(v2["a2"].unit, None)
        self.assertClose(v2["a2"], self.catalog["a2"])

    def testArrayColumn(self):
        """Test that an array column appears as a 2-d array with the expected shape.
        """
        v = self.catalog.asAstropy(unviewable="skip")
        self.assertClose(v["a3"], self.catalog["a3"])

    def testFlagColumn(self):
        """Test that Flag columns can be viewed if copy=True or unviewable="copy".
        """
        v1 = self.catalog.asAstropy(unviewable="copy")
        self.assertClose(v1["a4"], self.catalog["a4"])
        v2 = self.catalog.asAstropy(copy=True)
        self.assertClose(v2["a4"], self.catalog["a4"])

    def testCoordColumn(self):
        """Test that Coord columns appears as a pair of columns with correct angle units.
        """
        v1 = self.catalog.asAstropy(cls=astropy.table.Table, unviewable="skip")
        self.assertClose(v1["a5_ra"], self.catalog["a5_ra"])
        self.assertEqual(v1["a5_ra"].unit, astropy.units.Unit("rad"))
        self.assertNotIsInstance(v1["a5_ra"], astropy.units.Quantity)
        self.assertClose(v1["a5_dec"], self.catalog["a5_dec"])
        self.assertEqual(v1["a5_dec"].unit, astropy.units.Unit("rad"))
        self.assertNotIsInstance(v1["a5_dec"], astropy.units.Quantity)
        v2 = self.catalog.asAstropy(cls=astropy.table.QTable, unviewable="skip")
        self.assertClose(v2["a5_ra"]/astropy.units.Quantity(self.catalog["a5_ra"], unit="rad"), 1.0)
        self.assertEqual(v2["a5_ra"].unit, astropy.units.Unit("rad"))
        self.assertIsInstance(v2["a5_ra"], astropy.units.Quantity)
        self.assertClose(v2["a5_dec"]/astropy.units.Quantity(self.catalog["a5_dec"], unit="rad"), 1.0)
        self.assertEqual(v2["a5_dec"].unit, astropy.units.Unit("rad"))
        self.assertIsInstance(v2["a5_dec"], astropy.units.Quantity)

    def testStringColumn(self):
        """Test that string columns can be viewed if copy=True or unviewable='copy'.
        """
        v1 = self.catalog.asAstropy(unviewable="copy")
        self.assertEqual(v1["a6"][0], self.data[0]["a6"])
        self.assertEqual(v1["a6"][1], self.data[1]["a6"])
        v2 = self.catalog.asAstropy(copy=True)
        self.assertEqual(v2["a6"][0], self.data[0]["a6"])
        self.assertEqual(v2["a6"][1], self.data[1]["a6"])

    def testRaiseOnUnviewable(self):
        """Test that we can't view this table without copying, since it has Flag and String columns.
        """
        self.assertRaises(ValueError, self.catalog.asAstropy, copy=False, unviewable="raise")

    def testNoUnnecessaryCopies(self):
        """Test that fields that aren't Flag or String are not copied when copy=False (the default).
        """
        v1 = self.catalog.asAstropy(unviewable="copy")
        v1["a2"][0] = 4
        self.assertEqual(self.catalog[0]["a2"], 4)
        v2 = self.catalog.asAstropy(unviewable="skip")
        v2["a2"][1] = 10
        self.assertEqual(self.catalog[1]["a2"], 10)

    def testUnviewableSkip(self):
        """Test that we can skip unviewable columns.
        """
        v1 = self.catalog.asAstropy(unviewable="skip")
        self.assertRaises(KeyError, operator.getitem, v1, "a4")
        self.assertRaises(KeyError, operator.getitem, v1, "a6")


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass

def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()