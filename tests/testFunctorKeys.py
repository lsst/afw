#!/usr/bin/env python

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

import sys
import os
import unittest
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

numpy.random.seed(5)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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

    def doTestPointKey(self, fieldType, functorKeyType, valueType):
        schema = lsst.afw.table.Schema();
        schema.setVersion(0)
        xKey = schema.addField("a.x", type=fieldType, doc="x")
        yKey = schema.addField("a.y", type=fieldType, doc="y")
        # we create two equivalent functor keys, using the two different constructors
        fKey1 = functorKeyType(xKey, yKey)
        fKey2 = functorKeyType(schema["a"])
        # test that they're equivalent, and that their constituent keys are what we expect
        self.assertEqual(fKey1.getX(), xKey)
        self.assertEqual(fKey2.getX(), xKey)
        self.assertEqual(fKey1.getY(), yKey)
        self.assertEqual(fKey2.getY(), yKey)
        self.assertEqual(fKey1, fKey2)
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

    def testQuadrupoleKey(self):
        schema = lsst.afw.table.Schema();
        schema.setVersion(0)
        xxKey = schema.addField("a.xx", type=float, doc="xx")
        yyKey = schema.addField("a.yy", type=float, doc="yy")
        xyKey = schema.addField("a.xy", type=float, doc="xy")
        # we create two equivalent functor keys, using the two different constructors
        fKey1 = lsst.afw.table.QuadrupoleKey(xxKey, yyKey, xyKey)
        fKey2 = lsst.afw.table.QuadrupoleKey(schema["a"])
        # test that they're equivalent, and that their constituent keys are what we expect
        self.assertEqual(fKey1.getIxx(), xxKey)
        self.assertEqual(fKey2.getIxx(), xxKey)
        self.assertEqual(fKey1.getIyy(), yyKey)
        self.assertEqual(fKey2.getIyy(), yyKey)
        self.assertEqual(fKey1.getIxy(), xyKey)
        self.assertEqual(fKey2.getIxy(), xyKey)
        self.assertEqual(fKey1, fKey2)
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

    def doTestCovarianceMatrixKey(self, fieldType, parameterNames, varianceOnly, dynamicSize):
        schema = lsst.afw.table.Schema()
        schema.setVersion(0)
        sigmaKeys = []
        covKeys = []
        # we generate a schema with a complete set of fields for the diagonal and some (but not all)
        # of the covariance elements
        for i, pi in enumerate(parameterNames):
            sigmaKeys.append(schema.addField("a.%sSigma" % pi, type=fieldType, doc="uncertainty on %s" % pi))
            if varianceOnly:
                continue  # in this case we have fields for only the diagonal
            for pj in parameterNames[:i]:
                # intentionally be inconsistent about whether we store the lower or upper triangle,
                # and occasionally don't store anything at all; this tests that the
                # CovarianceMatrixKey constructor can handle all those possibilities.
                r = numpy.random.rand()
                if r < 0.3:
                    k = schema.addField("a.%s_%s_Cov" % (pi, pj), type=fieldType,
                                        doc="%s,%s covariance" % (pi, pj))
                elif r < 0.6:
                    k = schema.addField("a.%s_%s_Cov" % (pj, pi), type=fieldType,
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
            if varianceOnly: continue
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
            self.assertClose(matrix1[i,i], (i+1)*10 + (i+1), rtol=1E-7)
            if varianceOnly: continue
            for j in range(i):
                if covKeys[k].isValid():
                    self.assertClose(matrix1[i,j], (i+1)*10 + (j+1), rtol=1E-7)
                    self.assertClose(matrix2[i,j], (i+1)*10 + (j+1), rtol=1E-7)
                    self.assertClose(matrix1[j,i], (i+1)*10 + (j+1), rtol=1E-7)
                    self.assertClose(matrix2[j,i], (i+1)*10 + (j+1), rtol=1E-7)
                k += 1

    def testCovarianceMatrixKey(self):
        for fieldType in ("F", "D"):
            for parameterNames in (["x", "y"], ["xx", "yy", "xy"]):
                for varianceOnly in (True, False):
                    for dynamicSize in (True, False):
                        self.doTestCovarianceMatrixKey(fieldType, parameterNames, varianceOnly, dynamicSize)

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
        k1 = FunctorKeyType([a0, a1, a2])
        k2 = FunctorKeyType(schema["a"])
        k3 = FunctorKeyType(c)
        self.assertTrue(k1.isValid())
        self.assertTrue(k2.isValid())
        self.assertTrue(k3.isValid())
        self.assertEqual(k1, k2)
        self.assertEqual(k1[2], a2)
        self.assertEqual(k1[1:3], FunctorKeyType([a1, a2]))
        self.assertEqual(k1.getSize(), 3)
        self.assertEqual(k2.getSize(), 3)
        self.assertEqual(k3.getSize(), 4)
        self.assertNotEqual(k1, k3)
        self.assertRaises(IndexError, lambda k: k[1:3:2], k1)
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, FunctorKeyType, schema["b"])
        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, FunctorKeyType, [b0, b1, b2])
        table = lsst.afw.table.BaseTable.make(schema)
        record = table.makeRecord()
        array = numpy.random.randn(3).astype(numpyType)
        record.set(k1, array)
        self.assertClose(record.get(k1), array)
        self.assertClose(record.get(k2), array)
        self.assertEqual(record.get(k1).dtype, numpy.dtype(numpyType))

    def testArrayKey(self):
        self.doTestArrayKey("F", numpy.float32)
        self.doTestArrayKey("D", numpy.float64)

    def testCompoundKeyConverters(self):
        """Test that FunctorKeys that convert from old-style compound Keys work
        """
        schema = lsst.afw.table.Schema()

        # Create a schema full of old-style Keys
        pi1 = schema.addField("pi", type="PointI", doc="old-style Point field")
        pd1 = schema.addField("pd", type="PointD", doc="old-style Point field")
        q1 = schema.addField("q", type="MomentsD", doc="old-style Moments field")
        c1 = schema.addField("cov", type="CovF", doc="old-style Covariance field", size=4)
        cp1 = schema.addField("cov_p", type="CovPointF", doc="old-style Covariance<Point> field")
        cq1 = schema.addField("cov_q", type="CovMomentsF", doc="old-style Covariance<Moments> field")

        # Create FunctorKeys from the old-style Keys
        pi2 = lsst.afw.table.Point2IKey(pi1)
        pd2 = lsst.afw.table.Point2DKey(pd1)
        q2 = lsst.afw.table.QuadrupoleKey(q1)
        c2 = lsst.afw.table.makeCovarianceMatrixKey(c1)
        cp2 = lsst.afw.table.makeCovarianceMatrixKey(cp1)
        cq2 = lsst.afw.table.makeCovarianceMatrixKey(cq1)

        # Check that they're the same
        self.assertEqual(pi1.getX(), pi2.getX())
        self.assertEqual(pi1.getY(), pi2.getY())
        self.assertEqual(pd1.getX(), pd2.getX())
        self.assertEqual(pd1.getY(), pd2.getY())
        self.assertEqual(q1.getIxx(), q2.getIxx())
        self.assertEqual(q1.getIyy(), q2.getIyy())
        self.assertEqual(q1.getIxy(), q2.getIxy())

        # Covariance matrices are a little trickier; actually try getting/setting records to compare
        table = lsst.afw.table.BaseTable.make(schema)
        record1 = table.makeRecord()
        record2 = table.makeRecord()

        matrix = makePositiveSymmetricMatrix(4)
        record1.set(c1, matrix)
        self.assertClose(record1.get(c2), matrix)
        record2.set(c2, matrix)
        self.assertClose(record2.get(c1), matrix)

        matrix = makePositiveSymmetricMatrix(2)
        record1.set(cp1, matrix)
        self.assertClose(record1.get(cp2), matrix)
        record2.set(cp2, matrix)
        self.assertClose(record2.get(cp1), matrix)

        matrix = makePositiveSymmetricMatrix(3)
        record1.set(cq1, matrix)
        self.assertClose(record1.get(cq2), matrix)
        record2.set(cq2, matrix)
        self.assertClose(record2.get(cq1), matrix)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(FunctorKeysTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
