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

"""
Tests for table FunctorKeys

Run with:
   ./testFunctorKeys.py
or
   python
   >>> import testFunctorKeys; testFunctorKeys.run()
"""
from __future__ import absolute_import, division, print_function
import unittest

from builtins import range
import numpy

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.table
import lsst.afw.geom
import lsst.afw.coord

try:
    type(display)
except NameError:
    display = False


def makePositiveSymmetricMatrix(size):
    """Return a random symmetric matrix with only positive eigenvalues, suitable
    for use as a covariance matrix.
    """
    a = numpy.random.randn(size, size+1).astype(numpy.float32)
    m = numpy.dot(a, a.transpose())
    for i in range(size):
        for j in range(i):
            m[i, j] = m[j, i]
    return m


class FunctorKeysTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        numpy.random.seed(5)

    def doTestPointKey(self, fieldType, functorKeyType, valueType):
        schema = lsst.afw.table.Schema()
        fKey0 = functorKeyType.addFields(schema, "a", "x or y", "pixel")
        xKey = schema.find("a_x").key
        yKey = schema.find("a_y").key
        # we create two equivalent functor keys, using the two different constructors
        fKey1 = functorKeyType(xKey, yKey)
        fKey2 = functorKeyType(schema["a"])
        # test that they're equivalent, and that their constituent keys are what we expect
        self.assertEqual(fKey0.getX(), xKey)
        self.assertEqual(fKey0.getY(), yKey)
        self.assertEqual(fKey1.getX(), xKey)
        self.assertEqual(fKey2.getX(), xKey)
        self.assertEqual(fKey1.getY(), yKey)
        self.assertEqual(fKey2.getY(), yKey)
        self.assertEqual(fKey0, fKey1)
        self.assertEqual(fKey1, fKey2)
        self.assertTrue(fKey0.isValid())
        self.assertTrue(fKey1.isValid())
        self.assertTrue(fKey2.isValid())
        # check that a default-constructed functor key is invalid
        fKey3 = functorKeyType()
        self.assertNotEqual(fKey3, fKey1)
        self.assertFalse(fKey3.isValid())
        # create a record from the test schema, and fill it using the constituent keys
        table = lsst.afw.table.BaseTable.make(schema)
        record = table.makeRecord()
        record.set(xKey, 4)
        record.set(yKey, 2)
        # test that the return type and value is correct
        self.assertIsInstance(record.get(fKey1), valueType)
        self.assertEqual(record.get(fKey1).getX(), record.get(xKey))
        self.assertEqual(record.get(fKey1).getY(), record.get(yKey))
        # test that we can set using the functor key
        p = valueType(8, 16)
        record.set(fKey1, p)
        self.assertEqual(record.get(xKey), p.getX())
        self.assertEqual(record.get(yKey), p.getY())

    def testPointKey(self):
        self.doTestPointKey("I", lsst.afw.table.Point2IKey, lsst.afw.geom.Point2I)
        self.doTestPointKey("D", lsst.afw.table.Point2DKey, lsst.afw.geom.Point2D)

    def testCoordKey(self):
        schema = lsst.afw.table.Schema()
        fKey0 = lsst.afw.table.CoordKey.addFields(schema, "a", "position")
        longKey = schema.find("a_ra").key
        latKey = schema.find("a_dec").key
        # create two equivalent functor keys using the two different constructors
        fKey1 = lsst.afw.table.CoordKey(longKey, latKey)
        fKey2 = lsst.afw.table.CoordKey(schema["a"])
        # test that they are equivalent
        self.assertEqual(fKey0.getRa(), longKey)
        self.assertEqual(fKey0.getRa(), fKey1.getRa())
        self.assertEqual(fKey0.getRa(), fKey2.getRa())
        self.assertEqual(fKey0.getDec(), latKey)
        self.assertEqual(fKey0.getDec(), fKey1.getDec())
        self.assertEqual(fKey0.getDec(), fKey2.getDec())
        self.assertEqual(fKey0, fKey1)
        self.assertEqual(fKey0, fKey2)
        self.assertEqual(fKey1, fKey2)
        # a default-constructed key is invalid
        fKey3 = lsst.afw.table.CoordKey()
        self.assertFalse(fKey3.isValid())
        # create a record from the test schema, and fill it using the constituent keys
        table = lsst.afw.table.BaseTable.make(schema)
        record = table.makeRecord()
        record.set(longKey, lsst.afw.geom.Angle(0))
        record.set(latKey, lsst.afw.geom.Angle(1))
        self.assertIsInstance(record.get(fKey1), lsst.afw.coord.IcrsCoord)
        self.assertEqual(record.get(fKey1).getRa(), record.get(longKey))
        self.assertEqual(record.get(fKey1).getDec(), record.get(latKey))
        # Test that we can set using the functor key
        coord = lsst.afw.coord.IcrsCoord(lsst.afw.geom.Angle(0), lsst.afw.geom.Angle(1))
        record.set(fKey1, coord)
        self.assertEqual(record.get(longKey), coord.getRa())
        self.assertEqual(record.get(latKey), coord.getDec())
        # Check for inequality with a different key
        fKey3 = lsst.afw.table.CoordKey.addFields(schema, "b", "position")
        self.assertNotEqual(fKey0, fKey3)
        # test that we can assign a non-ICRS coordinate
        coord = lsst.afw.coord.Coord("11:11:11", "22:22:22", 1950)
        record.set(fKey0, coord)
        self.assertNotEqual(coord.getLongitude(), record.get(fKey0).getRa())
        self.assertEqual(coord.toIcrs().getRa(), record.get(fKey0).getRa())
        self.assertNotEqual(coord.getLatitude(), record.get(fKey0).getDec())
        self.assertEqual(coord.toIcrs().getDec(), record.get(fKey0).getDec())

    def testQuadrupoleKey(self):
        schema = lsst.afw.table.Schema()
        fKey0 = lsst.afw.table.QuadrupoleKey.addFields(
            schema, "a", "moments", lsst.afw.table.CoordinateType.PIXEL)
        xxKey = schema.find("a_xx").key
        yyKey = schema.find("a_yy").key
        xyKey = schema.find("a_xy").key
        # we create two equivalent functor keys, using the two different constructors
        fKey1 = lsst.afw.table.QuadrupoleKey(xxKey, yyKey, xyKey)
        fKey2 = lsst.afw.table.QuadrupoleKey(schema["a"])
        # test that they're equivalent, and tha=t their constituent keys are what we expect
        self.assertEqual(fKey0.getIxx(), xxKey)
        self.assertEqual(fKey1.getIxx(), xxKey)
        self.assertEqual(fKey2.getIxx(), xxKey)
        self.assertEqual(fKey0.getIyy(), yyKey)
        self.assertEqual(fKey1.getIyy(), yyKey)
        self.assertEqual(fKey2.getIyy(), yyKey)
        self.assertEqual(fKey0.getIxy(), xyKey)
        self.assertEqual(fKey1.getIxy(), xyKey)
        self.assertEqual(fKey2.getIxy(), xyKey)
        self.assertEqual(fKey0, fKey1)
        self.assertEqual(fKey1, fKey2)
        self.assertTrue(fKey0.isValid())
        self.assertTrue(fKey1.isValid())
        self.assertTrue(fKey2.isValid())
        # check that a default-constructed functor key is invalid
        fKey3 = lsst.afw.table.QuadrupoleKey()
        self.assertNotEqual(fKey3, fKey1)
        self.assertFalse(fKey3.isValid())
        # create a record from the test schema, and fill it using the constituent keys
        table = lsst.afw.table.BaseTable.make(schema)
        record = table.makeRecord()
        record.set(xxKey, 4)
        record.set(yyKey, 2)
        record.set(xyKey, 2)
        # test that the return type and value is correct
        self.assertIsInstance(record.get(fKey1), lsst.afw.geom.ellipses.Quadrupole)
        self.assertEqual(record.get(fKey1).getIxx(), record.get(xxKey))
        self.assertEqual(record.get(fKey1).getIyy(), record.get(yyKey))
        self.assertEqual(record.get(fKey1).getIxy(), record.get(xyKey))
        # test that we can set using the functor key
        p = lsst.afw.geom.ellipses.Quadrupole(8, 16, 4)
        record.set(fKey1, p)
        self.assertEqual(record.get(xxKey), p.getIxx())
        self.assertEqual(record.get(yyKey), p.getIyy())
        self.assertEqual(record.get(xyKey), p.getIxy())

    def testEllipseKey(self):
        schema = lsst.afw.table.Schema()
        fKey0 = lsst.afw.table.EllipseKey.addFields(schema, "a", "ellipse", "pixel")
        qKey = lsst.afw.table.QuadrupoleKey(schema["a"])
        pKey = lsst.afw.table.Point2DKey(schema["a"])
        # we create two more equivalent functor keys, using the two different constructors
        fKey1 = lsst.afw.table.EllipseKey(qKey, pKey)
        fKey2 = lsst.afw.table.EllipseKey(schema["a"])
        # test that they're equivalent, and tha=t their constituent keys are what we expect
        self.assertEqual(fKey0.getCore(), qKey)
        self.assertEqual(fKey1.getCore(), qKey)
        self.assertEqual(fKey2.getCore(), qKey)
        self.assertEqual(fKey0.getCenter(), pKey)
        self.assertEqual(fKey1.getCenter(), pKey)
        self.assertEqual(fKey2.getCenter(), pKey)
        self.assertEqual(fKey0, fKey1)
        self.assertEqual(fKey1, fKey2)
        self.assertTrue(fKey0.isValid())
        self.assertTrue(fKey1.isValid())
        self.assertTrue(fKey2.isValid())
        # check that a default-constructed functor key is invalid
        fKey3 = lsst.afw.table.EllipseKey()
        self.assertNotEqual(fKey3, fKey1)
        self.assertFalse(fKey3.isValid())
        # create a record from the test schema, and fill it using the constituent keys
        table = lsst.afw.table.BaseTable.make(schema)
        record = table.makeRecord()
        record.set(qKey, lsst.afw.geom.ellipses.Quadrupole(4, 3, 1))
        record.set(pKey, lsst.afw.geom.Point2D(5, 6))
        # test that the return type and value is correct
        self.assertIsInstance(record.get(fKey1), lsst.afw.geom.ellipses.Ellipse)
        self.assertClose(record.get(fKey1).getCore().getIxx(), record.get(qKey).getIxx(), rtol=1E-14)
        self.assertClose(record.get(fKey1).getCore().getIyy(), record.get(qKey).getIyy(), rtol=1E-14)
        self.assertClose(record.get(fKey1).getCore().getIxy(), record.get(qKey).getIxy(), rtol=1E-14)
        self.assertEqual(record.get(fKey1).getCenter().getX(), record.get(pKey).getX())
        self.assertEqual(record.get(fKey1).getCenter().getX(), record.get(pKey).getX())
        # test that we can set using the functor key
        e = lsst.afw.geom.ellipses.Ellipse(lsst.afw.geom.ellipses.Quadrupole(8, 16, 4),
                                           lsst.afw.geom.Point2D(5, 6))
        record.set(fKey1, e)
        self.assertClose(record.get(fKey1).getCore().getIxx(), e.getCore().getIxx(), rtol=1E-14)
        self.assertClose(record.get(fKey1).getCore().getIyy(), e.getCore().getIyy(), rtol=1E-14)
        self.assertClose(record.get(fKey1).getCore().getIxy(), e.getCore().getIxy(), rtol=1E-14)
        self.assertEqual(record.get(fKey1).getCenter().getX(), e.getCenter().getX())
        self.assertEqual(record.get(fKey1).getCenter().getX(), e.getCenter().getX())

    def doTestCovarianceMatrixKeyAddFields(self, fieldType, varianceOnly, dynamicSize):
        names = ["x", "y"]
        schema = lsst.afw.table.Schema()
        if dynamicSize:
            FunctorKeyType = getattr(lsst.afw.table, "CovarianceMatrix2%sKey" % fieldType.lower())
        else:
            FunctorKeyType = getattr(lsst.afw.table, "CovarianceMatrixX%sKey" % fieldType.lower())
        fKey1 = FunctorKeyType.addFields(schema, "a", names, ["m", "s"], varianceOnly)
        fKey2 = FunctorKeyType.addFields(schema, "b", names, "kg", varianceOnly)
        self.assertEqual(schema.find("a_xSigma").field.getUnits(), "m")
        self.assertEqual(schema.find("a_ySigma").field.getUnits(), "s")
        self.assertEqual(schema.find("b_xSigma").field.getUnits(), "kg")
        self.assertEqual(schema.find("b_ySigma").field.getUnits(), "kg")
        dtype = numpy.float64 if fieldType == "D" else numpy.float32
        if varianceOnly:
            m = numpy.diagflat(numpy.random.randn(2)**2).astype(dtype)
        else:
            self.assertEqual(schema.find("a_x_y_Cov").field.getUnits(), "m s")
            self.assertEqual(schema.find("b_x_y_Cov").field.getUnits(), "kg kg")
            v = numpy.random.randn(2, 2).astype(dtype)
            m = numpy.dot(v.transpose(), v)
        table = lsst.afw.table.BaseTable.make(schema)
        record = table.makeRecord()
        record.set(fKey1, m)
        self.assertClose(record.get(fKey1), m, rtol=1E-6)
        record.set(fKey2, m*2)
        self.assertClose(record.get(fKey2), m*2, rtol=1E-6)

    def doTestCovarianceMatrixKey(self, fieldType, parameterNames, varianceOnly, dynamicSize):
        schema = lsst.afw.table.Schema()
        sigmaKeys = []
        covKeys = []
        # we generate a schema with a complete set of fields for the diagonal and some (but not all)
        # of the covariance elements
        for i, pi in enumerate(parameterNames):
            sigmaKeys.append(schema.addField("a_%sSigma" % pi, type=fieldType, doc="uncertainty on %s" % pi))
            if varianceOnly:
                continue  # in this case we have fields for only the diagonal
            for pj in parameterNames[:i]:
                # intentionally be inconsistent about whether we store the lower or upper triangle,
                # and occasionally don't store anything at all; this tests that the
                # CovarianceMatrixKey constructor can handle all those possibilities.
                r = numpy.random.rand()
                if r < 0.3:
                    k = schema.addField("a_%s_%s_Cov" % (pi, pj), type=fieldType,
                                        doc="%s,%s covariance" % (pi, pj))
                elif r < 0.6:
                    k = schema.addField("a_%s_%s_Cov" % (pj, pi), type=fieldType,
                                        doc="%s,%s covariance" % (pj, pi))
                else:
                    k = lsst.afw.table.Key[fieldType]()
                covKeys.append(k)
        if dynamicSize:
            FunctorKeyType = getattr(lsst.afw.table, "CovarianceMatrix%d%sKey"
                                     % (len(parameterNames), fieldType.lower()))
        else:
            FunctorKeyType = getattr(lsst.afw.table, "CovarianceMatrixX%sKey"
                                     % fieldType.lower())
        # construct two equivalent functor keys using the different constructors
        fKey1 = FunctorKeyType(sigmaKeys, covKeys)
        fKey2 = FunctorKeyType(schema["a"], parameterNames)
        self.assertTrue(fKey1.isValid())
        self.assertTrue(fKey2.isValid())
        self.assertEqual(fKey1, fKey2)
        # verify that a default-constructed functor key is invalid
        fKey3 = FunctorKeyType()
        self.assertNotEqual(fKey3, fKey1)
        self.assertFalse(fKey3.isValid())
        # create a record from the test schema, and fill it using the constituent keys
        table = lsst.afw.table.BaseTable.make(schema)
        record = table.makeRecord()
        k = 0
        # we set each matrix element a two-digit number where the first digit is the row
        # index and the second digit is the column index.
        for i in range(len(parameterNames)):
            record.set(sigmaKeys[i], ((i+1)*10 + (i+1))**0.5)
            if varianceOnly:
                continue
            for j in range(i):
                if covKeys[k].isValid():
                    record.set(covKeys[k], (i+1)*10 + (j+1))
                k += 1
        # test that the return type and value is correct
        matrix1 = record.get(fKey1)
        matrix2 = record.get(fKey2)
        # we use assertClose because it can handle matrices, and because square root
        # in Python might not be exactly reversible with squaring in C++ (with possibly
        # different precision).
        self.assertClose(matrix1, matrix2)
        k = 0
        for i in range(len(parameterNames)):
            self.assertClose(matrix1[i, i], (i+1)*10 + (i+1), rtol=1E-7)
            if varianceOnly:
                continue
            for j in range(i):
                if covKeys[k].isValid():
                    self.assertClose(matrix1[i, j], (i+1)*10 + (j+1), rtol=1E-7)
                    self.assertClose(matrix2[i, j], (i+1)*10 + (j+1), rtol=1E-7)
                    self.assertClose(matrix1[j, i], (i+1)*10 + (j+1), rtol=1E-7)
                    self.assertClose(matrix2[j, i], (i+1)*10 + (j+1), rtol=1E-7)
                    self.assertClose(fKey1.getElement(record, i, j), (i+1)*10 + (j+1), rtol=1E-7)
                    self.assertClose(fKey2.getElement(record, i, j), (i+1)*10 + (j+1), rtol=1E-7)
                    v = numpy.random.randn()
                    fKey1.setElement(record, i, j, v)
                    self.assertClose(fKey2.getElement(record, i, j), v, rtol=1E-7)
                    fKey2.setElement(record, i, j, (i+1)*10 + (j+1))
                else:
                    self.assertRaisesLsstCpp(lsst.pex.exceptions.LogicError,
                                             fKey1.setElement, record, i, j, 0.0)
                k += 1

    def testCovarianceMatrixKey(self):
        for fieldType in ("F", "D"):
            for varianceOnly in (True, False):
                for dynamicSize in (True, False):
                    for parameterNames in (["x", "y"], ["xx", "yy", "xy"]):
                        self.doTestCovarianceMatrixKey(fieldType, parameterNames, varianceOnly, dynamicSize)
                    self.doTestCovarianceMatrixKeyAddFields(fieldType, varianceOnly, dynamicSize)

    def doTestArrayKey(self, fieldType, numpyType):
        FunctorKeyType = getattr(lsst.afw.table, "Array%sKey" % fieldType)
        self.assertFalse(FunctorKeyType().isValid())
        schema = lsst.afw.table.Schema()
        a0 = schema.addField("a_0", type=fieldType, doc="valid array element")
        a1 = schema.addField("a_1", type=fieldType, doc="valid array element")
        a2 = schema.addField("a_2", type=fieldType, doc="valid array element")
        b0 = schema.addField("b_0", type=fieldType, doc="invalid out-of-order array element")
        b2 = schema.addField("b_2", type=fieldType, doc="invalid out-of-order array element")
        b1 = schema.addField("b_1", type=fieldType, doc="invalid out-of-order array element")
        c = schema.addField("c", type="Array%s" % fieldType, doc="old-style array", size=4)
        k1 = FunctorKeyType([a0, a1, a2])  # construct from a list of keys
        k2 = FunctorKeyType(schema["a"])   # construct from SubSchema
        k3 = FunctorKeyType(c)             # construct from old-style Key<Array<T>>
        k4 = FunctorKeyType.addFields(schema, "d", "doc for d", "barn", 4)
        k5 = FunctorKeyType.addFields(schema, "e", "doc for e %3.1f", "barn", [2.1, 2.2])
        self.assertTrue(k1.isValid())
        self.assertTrue(k2.isValid())
        self.assertTrue(k3.isValid())
        self.assertTrue(k4.isValid())
        self.assertTrue(k5.isValid())
        self.assertEqual(k1, k2)      # k1 and k2 point to the same underlying fields
        self.assertEqual(k1[2], a2)   # test that we can extract an element
        self.assertEqual(k1[1:3], FunctorKeyType([a1, a2]))  # test that we can slice ArrayKeys
        self.assertEqual(k1.getSize(), 3)
        self.assertEqual(k2.getSize(), 3)
        self.assertEqual(k3.getSize(), 4)
        self.assertEqual(k4.getSize(), 4)
        self.assertEqual(k5.getSize(), 2)
        self.assertNotEqual(k1, k3)   # none of these point to the same underlying fields;
        self.assertNotEqual(k1, k4)   # they should all be unequal
        self.assertNotEqual(k1, k5)
        self.assertEqual(schema.find(k5[0]).field.getDoc(), "doc for e 2.1")  # test that the fields we added
        self.assertEqual(schema.find(k5[1]).field.getDoc(), "doc for e 2.2")  # got the right docs
        self.assertRaises(IndexError, lambda k: k[1:3:2], k1)  # test that invalid slices raise exceptions
        # test that trying to construct from a SubSchema with badly ordered fields doesn't work
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, FunctorKeyType, schema["b"])
        # test that trying to construct from a list of keys that are not ordered doesn't work
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, FunctorKeyType, [b0, b1, b2])
        self.assertEqual(k4, FunctorKeyType(schema["d"]))
        self.assertEqual(k5, FunctorKeyType(schema["e"]))
        # finally, we create a record, fill it with random data, and verify that get/set/__getitem__ work
        table = lsst.afw.table.BaseTable.make(schema)
        record = table.makeRecord()
        array = numpy.random.randn(3).astype(numpyType)
        record.set(k1, array)
        self.assertClose(record.get(k1), array)
        self.assertClose(record.get(k2), array)
        self.assertClose(record[k1], array)
        self.assertEqual(record.get(k1).dtype, numpy.dtype(numpyType))

    def testArrayKey(self):
        self.doTestArrayKey("F", numpy.float32)
        self.doTestArrayKey("D", numpy.float64)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
