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

class FunctorKeysTestCase(lsst.utils.tests.TestCase):

    def doTestPointKey(self, fieldType, functorKeyType, valueType):
        schema = lsst.afw.table.Schema();
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
