#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2016 AURA/LSST
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
#pybind11#"""
#pybind11#Tests for Astropy views into afw.table Catalogs
#pybind11#
#pybind11#Run with:
#pybind11#   ./testAstropyTableViews.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import testAstropyTableViews; testAstropyTableViews.run()
#pybind11#"""
#pybind11#
#pybind11#import unittest
#pybind11#import operator
#pybind11#
#pybind11#import numpy
#pybind11#import astropy.table
#pybind11#import astropy.units
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.table
#pybind11#
#pybind11#
#pybind11#class AstropyTableViewTestCase(lsst.utils.tests.TestCase):
#pybind11#    """Test that we can construct Astropy views to afw.table Catalog objects.
#pybind11#
#pybind11#    This test case does not yet test the syntax
#pybind11#
#pybind11#        table = astropy.table.Table(lsst_catalog)
#pybind11#
#pybind11#    which is made available by BaseCatalog.__astropy_table__, as this will not
#pybind11#    be available until Astropy 1.2 is released.  However, this simply
#pybind11#    delegates to BaseCatalog.asAstropy, which can also be called directly.
#pybind11#    """
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        self.k1 = schema.addField("a1", type=float, units="meter", doc="a1 (meter)")
#pybind11#        self.k2 = schema.addField("a2", type=int, doc="a2 (unitless)")
#pybind11#        self.k3 = schema.addField("a3", type="ArrayF", size=3, units="count", doc="a3 (array, counts)")
#pybind11#        self.k4 = schema.addField("a4", type="Flag", doc="a4 (flag)")
#pybind11#        self.k5 = lsst.afw.table.CoordKey.addFields(schema, "a5", "a5 coordinate")
#pybind11#        self.k6 = schema.addField("a6", type=str, size=8, doc="a6 (str)")
#pybind11#        self.catalog = lsst.afw.table.BaseCatalog(schema)
#pybind11#        self.data = [
#pybind11#            {
#pybind11#                "a1": 5.0, "a2": 3, "a3": numpy.array([0.5, 0.0, -0.5], dtype=numpy.float32),
#pybind11#                "a4": True, "a5_ra": 45.0*lsst.afw.geom.degrees, "a5_dec": 30.0*lsst.afw.geom.degrees,
#pybind11#                "a6": "bubbles"
#pybind11#            },
#pybind11#            {
#pybind11#                "a1": 2.5, "a2": 7, "a3": numpy.array([1.0, 0.5, -1.5], dtype=numpy.float32),
#pybind11#                "a4": False, "a5_ra": 25.0*lsst.afw.geom.degrees, "a5_dec": -60.0*lsst.afw.geom.degrees,
#pybind11#                "a6": "pingpong"
#pybind11#            },
#pybind11#        ]
#pybind11#        for d in self.data:
#pybind11#            record = self.catalog.addNew()
#pybind11#            for k, v in d.items():
#pybind11#                record.set(k, v)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.k1
#pybind11#        del self.k2
#pybind11#        del self.k3
#pybind11#        del self.k4
#pybind11#        del self.k5
#pybind11#        del self.k6
#pybind11#        del self.catalog
#pybind11#        del self.data
#pybind11#
#pybind11#    def testQuantityColumn(self):
#pybind11#        """Test that a column with units is handled as expected by Table and QTable.
#pybind11#        """
#pybind11#        v1 = self.catalog.asAstropy(cls=astropy.table.Table, unviewable="skip")
#pybind11#        self.assertEqual(v1["a1"].unit, astropy.units.Unit("m"))
#pybind11#        self.assertClose(v1["a1"], self.catalog["a1"])
#pybind11#        self.assertNotIsInstance(v1["a1"], astropy.units.Quantity)
#pybind11#        v2 = self.catalog.asAstropy(cls=astropy.table.QTable, unviewable="skip")
#pybind11#        self.assertEqual(v2["a1"].unit, astropy.units.Unit("m"))
#pybind11#        self.assertClose(v2["a1"]/astropy.units.Quantity(self.catalog["a1"]*100, "cm"), 1.0)
#pybind11#        self.assertIsInstance(v2["a1"], astropy.units.Quantity)
#pybind11#
#pybind11#    def testUnitlessColumn(self):
#pybind11#        """Test that a column without units is handled as expected by Table and QTable.
#pybind11#        """
#pybind11#        v1 = self.catalog.asAstropy(cls=astropy.table.Table, unviewable="skip")
#pybind11#        self.assertEqual(v1["a2"].unit, None)
#pybind11#        self.assertClose(v1["a2"], self.catalog["a2"])  # use assertClose just because it handles arrays
#pybind11#        v2 = self.catalog.asAstropy(cls=astropy.table.QTable, unviewable="skip")
#pybind11#        self.assertEqual(v2["a2"].unit, None)
#pybind11#        self.assertClose(v2["a2"], self.catalog["a2"])
#pybind11#
#pybind11#    def testArrayColumn(self):
#pybind11#        """Test that an array column appears as a 2-d array with the expected shape.
#pybind11#        """
#pybind11#        v = self.catalog.asAstropy(unviewable="skip")
#pybind11#        self.assertClose(v["a3"], self.catalog["a3"])
#pybind11#
#pybind11#    def testFlagColumn(self):
#pybind11#        """Test that Flag columns can be viewed if copy=True or unviewable="copy".
#pybind11#        """
#pybind11#        v1 = self.catalog.asAstropy(unviewable="copy")
#pybind11#        self.assertClose(v1["a4"], self.catalog["a4"])
#pybind11#        v2 = self.catalog.asAstropy(copy=True)
#pybind11#        self.assertClose(v2["a4"], self.catalog["a4"])
#pybind11#
#pybind11#    def testCoordColumn(self):
#pybind11#        """Test that Coord columns appears as a pair of columns with correct angle units.
#pybind11#        """
#pybind11#        v1 = self.catalog.asAstropy(cls=astropy.table.Table, unviewable="skip")
#pybind11#        self.assertClose(v1["a5_ra"], self.catalog["a5_ra"])
#pybind11#        self.assertEqual(v1["a5_ra"].unit, astropy.units.Unit("rad"))
#pybind11#        self.assertNotIsInstance(v1["a5_ra"], astropy.units.Quantity)
#pybind11#        self.assertClose(v1["a5_dec"], self.catalog["a5_dec"])
#pybind11#        self.assertEqual(v1["a5_dec"].unit, astropy.units.Unit("rad"))
#pybind11#        self.assertNotIsInstance(v1["a5_dec"], astropy.units.Quantity)
#pybind11#        v2 = self.catalog.asAstropy(cls=astropy.table.QTable, unviewable="skip")
#pybind11#        self.assertClose(v2["a5_ra"]/astropy.units.Quantity(self.catalog["a5_ra"], unit="rad"), 1.0)
#pybind11#        self.assertEqual(v2["a5_ra"].unit, astropy.units.Unit("rad"))
#pybind11#        self.assertIsInstance(v2["a5_ra"], astropy.units.Quantity)
#pybind11#        self.assertClose(v2["a5_dec"]/astropy.units.Quantity(self.catalog["a5_dec"], unit="rad"), 1.0)
#pybind11#        self.assertEqual(v2["a5_dec"].unit, astropy.units.Unit("rad"))
#pybind11#        self.assertIsInstance(v2["a5_dec"], astropy.units.Quantity)
#pybind11#
#pybind11#    def testStringColumn(self):
#pybind11#        """Test that string columns can be viewed if copy=True or unviewable='copy'.
#pybind11#        """
#pybind11#        v1 = self.catalog.asAstropy(unviewable="copy")
#pybind11#        self.assertEqual(v1["a6"][0], self.data[0]["a6"])
#pybind11#        self.assertEqual(v1["a6"][1], self.data[1]["a6"])
#pybind11#        v2 = self.catalog.asAstropy(copy=True)
#pybind11#        self.assertEqual(v2["a6"][0], self.data[0]["a6"])
#pybind11#        self.assertEqual(v2["a6"][1], self.data[1]["a6"])
#pybind11#
#pybind11#    def testRaiseOnUnviewable(self):
#pybind11#        """Test that we can't view this table without copying, since it has Flag and String columns.
#pybind11#        """
#pybind11#        self.assertRaises(ValueError, self.catalog.asAstropy, copy=False, unviewable="raise")
#pybind11#
#pybind11#    def testNoUnnecessaryCopies(self):
#pybind11#        """Test that fields that aren't Flag or String are not copied when copy=False (the default).
#pybind11#        """
#pybind11#        v1 = self.catalog.asAstropy(unviewable="copy")
#pybind11#        v1["a2"][0] = 4
#pybind11#        self.assertEqual(self.catalog[0]["a2"], 4)
#pybind11#        v2 = self.catalog.asAstropy(unviewable="skip")
#pybind11#        v2["a2"][1] = 10
#pybind11#        self.assertEqual(self.catalog[1]["a2"], 10)
#pybind11#
#pybind11#    def testUnviewableSkip(self):
#pybind11#        """Test that we can skip unviewable columns.
#pybind11#        """
#pybind11#        v1 = self.catalog.asAstropy(unviewable="skip")
#pybind11#        self.assertRaises(KeyError, operator.getitem, v1, "a4")
#pybind11#        self.assertRaises(KeyError, operator.getitem, v1, "a6")
#pybind11#
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
