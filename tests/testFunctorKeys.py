#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
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
#pybind11#Tests for table FunctorKeys
#pybind11#
#pybind11#Run with:
#pybind11#   ./testFunctorKeys.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import testFunctorKeys; testFunctorKeys.run()
#pybind11#"""
#pybind11#
#pybind11#import unittest
#pybind11#import numpy
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.afw.table
#pybind11#import lsst.afw.geom
#pybind11#import lsst.afw.coord
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11#def makePositiveSymmetricMatrix(size):
#pybind11#    """Return a random symmetric matrix with only positive eigenvalues, suitable
#pybind11#    for use as a covariance matrix.
#pybind11#    """
#pybind11#    a = numpy.random.randn(size, size+1).astype(numpy.float32)
#pybind11#    m = numpy.dot(a, a.transpose())
#pybind11#    for i in range(size):
#pybind11#        for j in range(i):
#pybind11#            m[i, j] = m[j, i]
#pybind11#    return m
#pybind11#
#pybind11#
#pybind11#class FunctorKeysTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        numpy.random.seed(5)
#pybind11#
#pybind11#    def doTestPointKey(self, fieldType, functorKeyType, valueType):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        fKey0 = functorKeyType.addFields(schema, "a", "x or y", "pixel")
#pybind11#        xKey = schema.find("a_x").key
#pybind11#        yKey = schema.find("a_y").key
#pybind11#        # we create two equivalent functor keys, using the two different constructors
#pybind11#        fKey1 = functorKeyType(xKey, yKey)
#pybind11#        fKey2 = functorKeyType(schema["a"])
#pybind11#        # test that they're equivalent, and that their constituent keys are what we expect
#pybind11#        self.assertEqual(fKey0.getX(), xKey)
#pybind11#        self.assertEqual(fKey0.getY(), yKey)
#pybind11#        self.assertEqual(fKey1.getX(), xKey)
#pybind11#        self.assertEqual(fKey2.getX(), xKey)
#pybind11#        self.assertEqual(fKey1.getY(), yKey)
#pybind11#        self.assertEqual(fKey2.getY(), yKey)
#pybind11#        self.assertEqual(fKey0, fKey1)
#pybind11#        self.assertEqual(fKey1, fKey2)
#pybind11#        self.assertTrue(fKey0.isValid())
#pybind11#        self.assertTrue(fKey1.isValid())
#pybind11#        self.assertTrue(fKey2.isValid())
#pybind11#        # check that a default-constructed functor key is invalid
#pybind11#        fKey3 = functorKeyType()
#pybind11#        self.assertNotEqual(fKey3, fKey1)
#pybind11#        self.assertFalse(fKey3.isValid())
#pybind11#        # create a record from the test schema, and fill it using the constituent keys
#pybind11#        table = lsst.afw.table.BaseTable.make(schema)
#pybind11#        record = table.makeRecord()
#pybind11#        record.set(xKey, 4)
#pybind11#        record.set(yKey, 2)
#pybind11#        # test that the return type and value is correct
#pybind11#        self.assertIsInstance(record.get(fKey1), valueType)
#pybind11#        self.assertEqual(record.get(fKey1).getX(), record.get(xKey))
#pybind11#        self.assertEqual(record.get(fKey1).getY(), record.get(yKey))
#pybind11#        # test that we can set using the functor key
#pybind11#        p = valueType(8, 16)
#pybind11#        record.set(fKey1, p)
#pybind11#        self.assertEqual(record.get(xKey), p.getX())
#pybind11#        self.assertEqual(record.get(yKey), p.getY())
#pybind11#
#pybind11#    def testPointKey(self):
#pybind11#        self.doTestPointKey("I", lsst.afw.table.Point2IKey, lsst.afw.geom.Point2I)
#pybind11#        self.doTestPointKey("D", lsst.afw.table.Point2DKey, lsst.afw.geom.Point2D)
#pybind11#
#pybind11#    def testCoordKey(self):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        fKey0 = lsst.afw.table.CoordKey.addFields(schema, "a", "position")
#pybind11#        longKey = schema.find("a_ra").key
#pybind11#        latKey = schema.find("a_dec").key
#pybind11#        # create two equivalent functor keys using the two different constructors
#pybind11#        fKey1 = lsst.afw.table.CoordKey(longKey, latKey)
#pybind11#        fKey2 = lsst.afw.table.CoordKey(schema["a"])
#pybind11#        # test that they are equivalent
#pybind11#        self.assertEqual(fKey0.getRa(), longKey)
#pybind11#        self.assertEqual(fKey0.getRa(), fKey1.getRa())
#pybind11#        self.assertEqual(fKey0.getRa(), fKey2.getRa())
#pybind11#        self.assertEqual(fKey0.getDec(), latKey)
#pybind11#        self.assertEqual(fKey0.getDec(), fKey1.getDec())
#pybind11#        self.assertEqual(fKey0.getDec(), fKey2.getDec())
#pybind11#        self.assertEqual(fKey0, fKey1)
#pybind11#        self.assertEqual(fKey0, fKey2)
#pybind11#        self.assertEqual(fKey1, fKey2)
#pybind11#        # a default-constructed key is invalid
#pybind11#        fKey3 = lsst.afw.table.CoordKey()
#pybind11#        self.assertFalse(fKey3.isValid())
#pybind11#        # create a record from the test schema, and fill it using the constituent keys
#pybind11#        table = lsst.afw.table.BaseTable.make(schema)
#pybind11#        record = table.makeRecord()
#pybind11#        record.set(longKey, lsst.afw.geom.Angle(0))
#pybind11#        record.set(latKey, lsst.afw.geom.Angle(1))
#pybind11#        self.assertIsInstance(record.get(fKey1), lsst.afw.coord.IcrsCoord)
#pybind11#        self.assertEqual(record.get(fKey1).getRa(), record.get(longKey))
#pybind11#        self.assertEqual(record.get(fKey1).getDec(), record.get(latKey))
#pybind11#        # Test that we can set using the functor key
#pybind11#        coord = lsst.afw.coord.IcrsCoord(lsst.afw.geom.Angle(0), lsst.afw.geom.Angle(1))
#pybind11#        record.set(fKey1, coord)
#pybind11#        self.assertEqual(record.get(longKey), coord.getRa())
#pybind11#        self.assertEqual(record.get(latKey), coord.getDec())
#pybind11#        # Check for inequality with a different key
#pybind11#        fKey3 = lsst.afw.table.CoordKey.addFields(schema, "b", "position")
#pybind11#        self.assertNotEqual(fKey0, fKey3)
#pybind11#        # test that we can assign a non-ICRS coordinate
#pybind11#        coord = lsst.afw.coord.Coord("11:11:11", "22:22:22", 1950)
#pybind11#        record.set(fKey0, coord)
#pybind11#        self.assertNotEqual(coord.getLongitude(), record.get(fKey0).getRa())
#pybind11#        self.assertEqual(coord.toIcrs().getRa(), record.get(fKey0).getRa())
#pybind11#        self.assertNotEqual(coord.getLatitude(), record.get(fKey0).getDec())
#pybind11#        self.assertEqual(coord.toIcrs().getDec(), record.get(fKey0).getDec())
#pybind11#
#pybind11#    def testQuadrupoleKey(self):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        fKey0 = lsst.afw.table.QuadrupoleKey.addFields(
#pybind11#            schema, "a", "moments", lsst.afw.table.CoordinateType_PIXEL)
#pybind11#        xxKey = schema.find("a_xx").key
#pybind11#        yyKey = schema.find("a_yy").key
#pybind11#        xyKey = schema.find("a_xy").key
#pybind11#        # we create two equivalent functor keys, using the two different constructors
#pybind11#        fKey1 = lsst.afw.table.QuadrupoleKey(xxKey, yyKey, xyKey)
#pybind11#        fKey2 = lsst.afw.table.QuadrupoleKey(schema["a"])
#pybind11#        # test that they're equivalent, and tha=t their constituent keys are what we expect
#pybind11#        self.assertEqual(fKey0.getIxx(), xxKey)
#pybind11#        self.assertEqual(fKey1.getIxx(), xxKey)
#pybind11#        self.assertEqual(fKey2.getIxx(), xxKey)
#pybind11#        self.assertEqual(fKey0.getIyy(), yyKey)
#pybind11#        self.assertEqual(fKey1.getIyy(), yyKey)
#pybind11#        self.assertEqual(fKey2.getIyy(), yyKey)
#pybind11#        self.assertEqual(fKey0.getIxy(), xyKey)
#pybind11#        self.assertEqual(fKey1.getIxy(), xyKey)
#pybind11#        self.assertEqual(fKey2.getIxy(), xyKey)
#pybind11#        self.assertEqual(fKey0, fKey1)
#pybind11#        self.assertEqual(fKey1, fKey2)
#pybind11#        self.assertTrue(fKey0.isValid())
#pybind11#        self.assertTrue(fKey1.isValid())
#pybind11#        self.assertTrue(fKey2.isValid())
#pybind11#        # check that a default-constructed functor key is invalid
#pybind11#        fKey3 = lsst.afw.table.QuadrupoleKey()
#pybind11#        self.assertNotEqual(fKey3, fKey1)
#pybind11#        self.assertFalse(fKey3.isValid())
#pybind11#        # create a record from the test schema, and fill it using the constituent keys
#pybind11#        table = lsst.afw.table.BaseTable.make(schema)
#pybind11#        record = table.makeRecord()
#pybind11#        record.set(xxKey, 4)
#pybind11#        record.set(yyKey, 2)
#pybind11#        record.set(xyKey, 2)
#pybind11#        # test that the return type and value is correct
#pybind11#        self.assertIsInstance(record.get(fKey1), lsst.afw.geom.ellipses.Quadrupole)
#pybind11#        self.assertEqual(record.get(fKey1).getIxx(), record.get(xxKey))
#pybind11#        self.assertEqual(record.get(fKey1).getIyy(), record.get(yyKey))
#pybind11#        self.assertEqual(record.get(fKey1).getIxy(), record.get(xyKey))
#pybind11#        # test that we can set using the functor key
#pybind11#        p = lsst.afw.geom.ellipses.Quadrupole(8, 16, 4)
#pybind11#        record.set(fKey1, p)
#pybind11#        self.assertEqual(record.get(xxKey), p.getIxx())
#pybind11#        self.assertEqual(record.get(yyKey), p.getIyy())
#pybind11#        self.assertEqual(record.get(xyKey), p.getIxy())
#pybind11#
#pybind11#    def testEllipseKey(self):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        fKey0 = lsst.afw.table.EllipseKey.addFields(schema, "a", "ellipse", "pixel")
#pybind11#        qKey = lsst.afw.table.QuadrupoleKey(schema["a"])
#pybind11#        pKey = lsst.afw.table.Point2DKey(schema["a"])
#pybind11#        # we create two more equivalent functor keys, using the two different constructors
#pybind11#        fKey1 = lsst.afw.table.EllipseKey(qKey, pKey)
#pybind11#        fKey2 = lsst.afw.table.EllipseKey(schema["a"])
#pybind11#        # test that they're equivalent, and tha=t their constituent keys are what we expect
#pybind11#        self.assertEqual(fKey0.getCore(), qKey)
#pybind11#        self.assertEqual(fKey1.getCore(), qKey)
#pybind11#        self.assertEqual(fKey2.getCore(), qKey)
#pybind11#        self.assertEqual(fKey0.getCenter(), pKey)
#pybind11#        self.assertEqual(fKey1.getCenter(), pKey)
#pybind11#        self.assertEqual(fKey2.getCenter(), pKey)
#pybind11#        self.assertEqual(fKey0, fKey1)
#pybind11#        self.assertEqual(fKey1, fKey2)
#pybind11#        self.assertTrue(fKey0.isValid())
#pybind11#        self.assertTrue(fKey1.isValid())
#pybind11#        self.assertTrue(fKey2.isValid())
#pybind11#        # check that a default-constructed functor key is invalid
#pybind11#        fKey3 = lsst.afw.table.EllipseKey()
#pybind11#        self.assertNotEqual(fKey3, fKey1)
#pybind11#        self.assertFalse(fKey3.isValid())
#pybind11#        # create a record from the test schema, and fill it using the constituent keys
#pybind11#        table = lsst.afw.table.BaseTable.make(schema)
#pybind11#        record = table.makeRecord()
#pybind11#        record.set(qKey, lsst.afw.geom.ellipses.Quadrupole(4, 3, 1))
#pybind11#        record.set(pKey, lsst.afw.geom.Point2D(5, 6))
#pybind11#        # test that the return type and value is correct
#pybind11#        self.assertIsInstance(record.get(fKey1), lsst.afw.geom.ellipses.Ellipse)
#pybind11#        self.assertClose(record.get(fKey1).getCore().getIxx(), record.get(qKey).getIxx(), rtol=1E-14)
#pybind11#        self.assertClose(record.get(fKey1).getCore().getIyy(), record.get(qKey).getIyy(), rtol=1E-14)
#pybind11#        self.assertClose(record.get(fKey1).getCore().getIxy(), record.get(qKey).getIxy(), rtol=1E-14)
#pybind11#        self.assertEqual(record.get(fKey1).getCenter().getX(), record.get(pKey).getX())
#pybind11#        self.assertEqual(record.get(fKey1).getCenter().getX(), record.get(pKey).getX())
#pybind11#        # test that we can set using the functor key
#pybind11#        e = lsst.afw.geom.ellipses.Ellipse(lsst.afw.geom.ellipses.Quadrupole(8, 16, 4),
#pybind11#                                           lsst.afw.geom.Point2D(5, 6))
#pybind11#        record.set(fKey1, e)
#pybind11#        self.assertClose(record.get(fKey1).getCore().getIxx(), e.getCore().getIxx(), rtol=1E-14)
#pybind11#        self.assertClose(record.get(fKey1).getCore().getIyy(), e.getCore().getIyy(), rtol=1E-14)
#pybind11#        self.assertClose(record.get(fKey1).getCore().getIxy(), e.getCore().getIxy(), rtol=1E-14)
#pybind11#        self.assertEqual(record.get(fKey1).getCenter().getX(), e.getCenter().getX())
#pybind11#        self.assertEqual(record.get(fKey1).getCenter().getX(), e.getCenter().getX())
#pybind11#
#pybind11#    def doTestCovarianceMatrixKeyAddFields(self, fieldType, varianceOnly, dynamicSize):
#pybind11#        names = ["x", "y"]
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        if dynamicSize:
#pybind11#            FunctorKeyType = getattr(lsst.afw.table, "CovarianceMatrix2%sKey" % fieldType.lower())
#pybind11#        else:
#pybind11#            FunctorKeyType = getattr(lsst.afw.table, "CovarianceMatrixX%sKey" % fieldType.lower())
#pybind11#        fKey1 = FunctorKeyType.addFields(schema, "a", names, ["m", "s"], varianceOnly)
#pybind11#        fKey2 = FunctorKeyType.addFields(schema, "b", names, "kg", varianceOnly)
#pybind11#        self.assertEqual(schema.find("a_xSigma").field.getUnits(), "m")
#pybind11#        self.assertEqual(schema.find("a_ySigma").field.getUnits(), "s")
#pybind11#        self.assertEqual(schema.find("b_xSigma").field.getUnits(), "kg")
#pybind11#        self.assertEqual(schema.find("b_ySigma").field.getUnits(), "kg")
#pybind11#        dtype = numpy.float64 if fieldType == "D" else numpy.float32
#pybind11#        if varianceOnly:
#pybind11#            m = numpy.diagflat(numpy.random.randn(2)**2).astype(dtype)
#pybind11#        else:
#pybind11#            self.assertEqual(schema.find("a_x_y_Cov").field.getUnits(), "m s")
#pybind11#            self.assertEqual(schema.find("b_x_y_Cov").field.getUnits(), "kg kg")
#pybind11#            v = numpy.random.randn(2, 2).astype(dtype)
#pybind11#            m = numpy.dot(v.transpose(), v)
#pybind11#        table = lsst.afw.table.BaseTable.make(schema)
#pybind11#        record = table.makeRecord()
#pybind11#        record.set(fKey1, m)
#pybind11#        self.assertClose(record.get(fKey1), m, rtol=1E-6)
#pybind11#        record.set(fKey2, m*2)
#pybind11#        self.assertClose(record.get(fKey2), m*2, rtol=1E-6)
#pybind11#
#pybind11#    def doTestCovarianceMatrixKey(self, fieldType, parameterNames, varianceOnly, dynamicSize):
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        sigmaKeys = []
#pybind11#        covKeys = []
#pybind11#        # we generate a schema with a complete set of fields for the diagonal and some (but not all)
#pybind11#        # of the covariance elements
#pybind11#        for i, pi in enumerate(parameterNames):
#pybind11#            sigmaKeys.append(schema.addField("a_%sSigma" % pi, type=fieldType, doc="uncertainty on %s" % pi))
#pybind11#            if varianceOnly:
#pybind11#                continue  # in this case we have fields for only the diagonal
#pybind11#            for pj in parameterNames[:i]:
#pybind11#                # intentionally be inconsistent about whether we store the lower or upper triangle,
#pybind11#                # and occasionally don't store anything at all; this tests that the
#pybind11#                # CovarianceMatrixKey constructor can handle all those possibilities.
#pybind11#                r = numpy.random.rand()
#pybind11#                if r < 0.3:
#pybind11#                    k = schema.addField("a_%s_%s_Cov" % (pi, pj), type=fieldType,
#pybind11#                                        doc="%s,%s covariance" % (pi, pj))
#pybind11#                elif r < 0.6:
#pybind11#                    k = schema.addField("a_%s_%s_Cov" % (pj, pi), type=fieldType,
#pybind11#                                        doc="%s,%s covariance" % (pj, pi))
#pybind11#                else:
#pybind11#                    k = lsst.afw.table.Key[fieldType]()
#pybind11#                covKeys.append(k)
#pybind11#        if dynamicSize:
#pybind11#            FunctorKeyType = getattr(lsst.afw.table, "CovarianceMatrix%d%sKey"
#pybind11#                                     % (len(parameterNames), fieldType.lower()))
#pybind11#        else:
#pybind11#            FunctorKeyType = getattr(lsst.afw.table, "CovarianceMatrixX%sKey"
#pybind11#                                     % fieldType.lower())
#pybind11#        # construct two equivalent functor keys using the different constructors
#pybind11#        fKey1 = FunctorKeyType(sigmaKeys, covKeys)
#pybind11#        fKey2 = FunctorKeyType(schema["a"], parameterNames)
#pybind11#        self.assertTrue(fKey1.isValid())
#pybind11#        self.assertTrue(fKey2.isValid())
#pybind11#        self.assertEqual(fKey1, fKey2)
#pybind11#        # verify that a default-constructed functor key is invalid
#pybind11#        fKey3 = FunctorKeyType()
#pybind11#        self.assertNotEqual(fKey3, fKey1)
#pybind11#        self.assertFalse(fKey3.isValid())
#pybind11#        # create a record from the test schema, and fill it using the constituent keys
#pybind11#        table = lsst.afw.table.BaseTable.make(schema)
#pybind11#        record = table.makeRecord()
#pybind11#        k = 0
#pybind11#        # we set each matrix element a two-digit number where the first digit is the row
#pybind11#        # index and the second digit is the column index.
#pybind11#        for i in range(len(parameterNames)):
#pybind11#            record.set(sigmaKeys[i], ((i+1)*10 + (i+1))**0.5)
#pybind11#            if varianceOnly:
#pybind11#                continue
#pybind11#            for j in range(i):
#pybind11#                if covKeys[k].isValid():
#pybind11#                    record.set(covKeys[k], (i+1)*10 + (j+1))
#pybind11#                k += 1
#pybind11#        # test that the return type and value is correct
#pybind11#        matrix1 = record.get(fKey1)
#pybind11#        matrix2 = record.get(fKey2)
#pybind11#        # we use assertClose because it can handle matrices, and because square root
#pybind11#        # in Python might not be exactly reversible with squaring in C++ (with possibly
#pybind11#        # different precision).
#pybind11#        self.assertClose(matrix1, matrix2)
#pybind11#        k = 0
#pybind11#        for i in range(len(parameterNames)):
#pybind11#            self.assertClose(matrix1[i, i], (i+1)*10 + (i+1), rtol=1E-7)
#pybind11#            if varianceOnly:
#pybind11#                continue
#pybind11#            for j in range(i):
#pybind11#                if covKeys[k].isValid():
#pybind11#                    self.assertClose(matrix1[i, j], (i+1)*10 + (j+1), rtol=1E-7)
#pybind11#                    self.assertClose(matrix2[i, j], (i+1)*10 + (j+1), rtol=1E-7)
#pybind11#                    self.assertClose(matrix1[j, i], (i+1)*10 + (j+1), rtol=1E-7)
#pybind11#                    self.assertClose(matrix2[j, i], (i+1)*10 + (j+1), rtol=1E-7)
#pybind11#                    self.assertClose(fKey1.getElement(record, i, j), (i+1)*10 + (j+1), rtol=1E-7)
#pybind11#                    self.assertClose(fKey2.getElement(record, i, j), (i+1)*10 + (j+1), rtol=1E-7)
#pybind11#                    v = numpy.random.randn()
#pybind11#                    fKey1.setElement(record, i, j, v)
#pybind11#                    self.assertClose(fKey2.getElement(record, i, j), v, rtol=1E-7)
#pybind11#                    fKey2.setElement(record, i, j, (i+1)*10 + (j+1))
#pybind11#                else:
#pybind11#                    self.assertRaisesLsstCpp(lsst.pex.exceptions.LogicError,
#pybind11#                                             fKey1.setElement, record, i, j, 0.0)
#pybind11#                k += 1
#pybind11#
#pybind11#    def testCovarianceMatrixKey(self):
#pybind11#        for fieldType in ("F", "D"):
#pybind11#            for varianceOnly in (True, False):
#pybind11#                for dynamicSize in (True, False):
#pybind11#                    for parameterNames in (["x", "y"], ["xx", "yy", "xy"]):
#pybind11#                        self.doTestCovarianceMatrixKey(fieldType, parameterNames, varianceOnly, dynamicSize)
#pybind11#                    self.doTestCovarianceMatrixKeyAddFields(fieldType, varianceOnly, dynamicSize)
#pybind11#
#pybind11#    def doTestArrayKey(self, fieldType, numpyType):
#pybind11#        FunctorKeyType = getattr(lsst.afw.table, "Array%sKey" % fieldType)
#pybind11#        self.assertFalse(FunctorKeyType().isValid())
#pybind11#        schema = lsst.afw.table.Schema()
#pybind11#        a0 = schema.addField("a_0", type=fieldType, doc="valid array element")
#pybind11#        a1 = schema.addField("a_1", type=fieldType, doc="valid array element")
#pybind11#        a2 = schema.addField("a_2", type=fieldType, doc="valid array element")
#pybind11#        b0 = schema.addField("b_0", type=fieldType, doc="invalid out-of-order array element")
#pybind11#        b2 = schema.addField("b_2", type=fieldType, doc="invalid out-of-order array element")
#pybind11#        b1 = schema.addField("b_1", type=fieldType, doc="invalid out-of-order array element")
#pybind11#        c = schema.addField("c", type="Array%s" % fieldType, doc="old-style array", size=4)
#pybind11#        k1 = FunctorKeyType([a0, a1, a2])  # construct from a vector of keys
#pybind11#        k2 = FunctorKeyType(schema["a"])   # construct from SubSchema
#pybind11#        k3 = FunctorKeyType(c)             # construct from old-style Key<Array<T>>
#pybind11#        k4 = FunctorKeyType.addFields(schema, "d", "doc for d", "barn", 4)
#pybind11#        k5 = FunctorKeyType.addFields(schema, "e", "doc for e %3.1f", "barn", [2.1, 2.2])
#pybind11#        self.assertTrue(k1.isValid())
#pybind11#        self.assertTrue(k2.isValid())
#pybind11#        self.assertTrue(k3.isValid())
#pybind11#        self.assertTrue(k4.isValid())
#pybind11#        self.assertTrue(k5.isValid())
#pybind11#        self.assertEqual(k1, k2)      # k1 and k2 point to the same underlying fields
#pybind11#        self.assertEqual(k1[2], a2)   # test that we can extract an element
#pybind11#        self.assertEqual(k1[1:3], FunctorKeyType([a1, a2]))  # test that we can slice ArrayKeys
#pybind11#        self.assertEqual(k1.getSize(), 3)
#pybind11#        self.assertEqual(k2.getSize(), 3)
#pybind11#        self.assertEqual(k3.getSize(), 4)
#pybind11#        self.assertEqual(k4.getSize(), 4)
#pybind11#        self.assertEqual(k5.getSize(), 2)
#pybind11#        self.assertNotEqual(k1, k3)   # none of these point to the same underlying fields;
#pybind11#        self.assertNotEqual(k1, k4)   # they should all be unequal
#pybind11#        self.assertNotEqual(k1, k5)
#pybind11#        self.assertEqual(schema.find(k5[0]).field.getDoc(), "doc for e 2.1")  # test that the fields we added
#pybind11#        self.assertEqual(schema.find(k5[1]).field.getDoc(), "doc for e 2.2")  # got the right docs
#pybind11#        self.assertRaises(IndexError, lambda k: k[1:3:2], k1)  # test that invalid slices raise exceptions
#pybind11#        # test that trying to construct from a SubSchema with badly ordered fields doesn't work
#pybind11#        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, FunctorKeyType, schema["b"])
#pybind11#        # test that trying to construct from a list of keys that are not ordered doesn't work
#pybind11#        self.assertRaises(lsst.pex.exceptions.InvalidParameterError, FunctorKeyType, [b0, b1, b2])
#pybind11#        self.assertEqual(k4, FunctorKeyType(schema["d"]))
#pybind11#        self.assertEqual(k5, FunctorKeyType(schema["e"]))
#pybind11#        # finally, we create a record, fill it with random data, and verify that get/set/__getitem__ work
#pybind11#        table = lsst.afw.table.BaseTable.make(schema)
#pybind11#        record = table.makeRecord()
#pybind11#        array = numpy.random.randn(3).astype(numpyType)
#pybind11#        record.set(k1, array)
#pybind11#        self.assertClose(record.get(k1), array)
#pybind11#        self.assertClose(record.get(k2), array)
#pybind11#        self.assertClose(record[k1], array)
#pybind11#        self.assertEqual(record.get(k1).dtype, numpy.dtype(numpyType))
#pybind11#
#pybind11#    def testArrayKey(self):
#pybind11#        self.doTestArrayKey("F", numpy.float32)
#pybind11#        self.doTestArrayKey("D", numpy.float64)
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
