#pybind11##!/usr/bin/env python
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
#pybind11#Tests for math.ChebyshevBoundedField
#pybind11#
#pybind11#Run with:
#pybind11#   ./testChebyshevBoundedField.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import testSchema; testSchema.run()
#pybind11#"""
#pybind11#from __future__ import division
#pybind11#from builtins import range
#pybind11#
#pybind11#import os
#pybind11#import unittest
#pybind11#import numpy
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.afw.geom
#pybind11#import lsst.afw.math
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11#CHEBYSHEV_T = [
#pybind11#    lambda x: x**0,
#pybind11#    lambda x: x,
#pybind11#    lambda x: 2*x**2 - 1,
#pybind11#    lambda x: (4*x**2 - 3)*x,
#pybind11#    lambda x: (8*x**2 - 8)*x**2 + 1,
#pybind11#    lambda x: ((16*x**2 - 20)*x**2 + 5)*x,
#pybind11#]
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class ChebyshevBoundedFieldTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        numpy.random.seed(5)
#pybind11#        self.bbox = lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(-5, -5), lsst.afw.geom.Point2I(5, 5))
#pybind11#        self.x1d = numpy.linspace(self.bbox.getBeginX(), self.bbox.getEndX())
#pybind11#        self.y1d = numpy.linspace(self.bbox.getBeginY(), self.bbox.getEndY())
#pybind11#        self.x2d, self.y2d = numpy.meshgrid(self.x1d, self.y1d)
#pybind11#        self.xFlat = numpy.ravel(self.x2d)
#pybind11#        self.yFlat = numpy.ravel(self.y2d)
#pybind11#        self.cases = []
#pybind11#        for orderX in range(0, 5):
#pybind11#            for orderY in range(0, 5):
#pybind11#                indexX, indexY = numpy.meshgrid(numpy.arange(orderX+1, dtype=int),
#pybind11#                                                numpy.arange(orderY+1, dtype=int))
#pybind11#                for triangular in (True, False):
#pybind11#                    ctrl = lsst.afw.math.ChebyshevBoundedFieldControl()
#pybind11#                    ctrl.orderX = orderX
#pybind11#                    ctrl.orderY = orderY
#pybind11#                    ctrl.triangular = triangular
#pybind11#                    coefficients = numpy.random.randn(orderY+1, orderX+1)
#pybind11#                    if triangular:
#pybind11#                        coefficients[indexX + indexY > max(orderX, orderY)] = 0.0
#pybind11#                    self.cases.append((ctrl, coefficients))
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.bbox
#pybind11#
#pybind11#    def testEvaluate(self):
#pybind11#        """Test the single-point evaluate method against explicitly-defined 1-d Chebyshevs
#pybind11#        (at the top of this file).
#pybind11#        """
#pybind11#        factor = 12.345
#pybind11#        boxD = lsst.afw.geom.Box2D(self.bbox)
#pybind11#        # sx, sy: transform from self.bbox range to [-1, -1]
#pybind11#        sx = 2.0 / boxD.getWidth()
#pybind11#        sy = 2.0 / boxD.getHeight()
#pybind11#        nPoints = 50
#pybind11#        for ctrl, coefficients in self.cases:
#pybind11#            field = lsst.afw.math.ChebyshevBoundedField(self.bbox, coefficients)
#pybind11#            x = numpy.random.rand(nPoints)*boxD.getWidth() + boxD.getMinX()
#pybind11#            y = numpy.random.rand(nPoints)*boxD.getHeight() + boxD.getMinY()
#pybind11#            z1 = field.evaluate(x, y)
#pybind11#            tx = numpy.array([CHEBYSHEV_T[i](sx*x) for i in range(coefficients.shape[1])])
#pybind11#            ty = numpy.array([CHEBYSHEV_T[i](sy*y) for i in range(coefficients.shape[0])])
#pybind11#            self.assertEqual(tx.shape, (coefficients.shape[1], x.size))
#pybind11#            self.assertEqual(ty.shape, (coefficients.shape[0], y.size))
#pybind11#            z2 = numpy.array([numpy.dot(ty[:, i], numpy.dot(coefficients, tx[:, i]))
#pybind11#                              for i in range(nPoints)])
#pybind11#            self.assertFloatsAlmostEqual(z1, z2, rtol=1E-12)
#pybind11#
#pybind11#            scaled = lsst.afw.math.ChebyshevBoundedField.cast(field*factor)
#pybind11#            self.assertFloatsAlmostEqual(scaled.evaluate(x, y), factor*z2, rtol=factor*1E-13)
#pybind11#            self.assertFloatsEqual(scaled.getCoefficients(), factor*field.getCoefficients())
#pybind11#
#pybind11#    def testImageFit(self):
#pybind11#        """Test that we can fit an image produced by a ChebyshevBoundedField and
#pybind11#        get the same coefficients back.
#pybind11#        """
#pybind11#        for ctrl, coefficients in self.cases:
#pybind11#            inField = lsst.afw.math.ChebyshevBoundedField(self.bbox, coefficients)
#pybind11#            for Image in (lsst.afw.image.ImageF, lsst.afw.image.ImageD):
#pybind11#                image = Image(self.bbox)
#pybind11#                inField.fillImage(image)
#pybind11#                outField = lsst.afw.math.ChebyshevBoundedField.fit(image, ctrl)
#pybind11#                self.assertFloatsAlmostEqual(outField.getCoefficients(), coefficients, rtol=1E-6, atol=1E-7)
#pybind11#
#pybind11#    def testArrayFit(self):
#pybind11#        """Test that we can fit 1-d arrays produced by a ChebyshevBoundedField and
#pybind11#        get the same coefficients back.
#pybind11#        """
#pybind11#        for ctrl, coefficients in self.cases:
#pybind11#            inField = lsst.afw.math.ChebyshevBoundedField(self.bbox, coefficients)
#pybind11#            for Image in (lsst.afw.image.ImageF, lsst.afw.image.ImageD):
#pybind11#                array = inField.evaluate(self.xFlat, self.yFlat)
#pybind11#                outField1 = lsst.afw.math.ChebyshevBoundedField.fit(self.bbox, self.xFlat, self.yFlat,
#pybind11#                                                                    array, ctrl)
#pybind11#                self.assertFloatsAlmostEqual(outField1.getCoefficients(), coefficients, rtol=1E-6, atol=1E-7)
#pybind11#                weights = (1.0 + numpy.random.randn(array.size)**2)
#pybind11#                # Should get same results with different weights, since we still have no noise
#pybind11#                # and a model that can exactly reproduce the data.
#pybind11#                outField2 = lsst.afw.math.ChebyshevBoundedField.fit(self.bbox, self.xFlat, self.yFlat,
#pybind11#                                                                    array, weights, ctrl)
#pybind11#                self.assertFloatsAlmostEqual(outField2.getCoefficients(), coefficients, rtol=1E-7, atol=1E-7)
#pybind11#
#pybind11#    def testPersistence(self):
#pybind11#        """Test that we can fit 1-d arrays produced by a ChebyshevBoundedField and
#pybind11#        get the same coefficients back.
#pybind11#        """
#pybind11#        filename = "testChebyshevBoundedField.fits"
#pybind11#        boxD = lsst.afw.geom.Box2D(self.bbox)
#pybind11#        nPoints = 50
#pybind11#        for ctrl, coefficients in self.cases:
#pybind11#            inField = lsst.afw.math.ChebyshevBoundedField(self.bbox, coefficients)
#pybind11#            inField.writeFits(filename)
#pybind11#            outField = lsst.afw.math.ChebyshevBoundedField.readFits(filename)
#pybind11#            self.assertEqual(inField.getBBox(), outField.getBBox())
#pybind11#            self.assertFloatsAlmostEqual(inField.getCoefficients(), outField.getCoefficients())
#pybind11#            x = numpy.random.rand(nPoints)*boxD.getWidth() + boxD.getMinX()
#pybind11#            y = numpy.random.rand(nPoints)*boxD.getHeight() + boxD.getMinY()
#pybind11#            z1 = inField.evaluate(x, y)
#pybind11#            z2 = inField.evaluate(x, y)
#pybind11#            self.assertFloatsAlmostEqual(z1, z2, rtol=1E-13)
#pybind11#        os.remove(filename)
#pybind11#
#pybind11#    def testTruncate(self):
#pybind11#        """Test that truncate() works as expected
#pybind11#        """
#pybind11#        for ctrl, coefficients in self.cases:
#pybind11#            field1 = lsst.afw.math.ChebyshevBoundedField(self.bbox, coefficients)
#pybind11#            field2 = field1.truncate(ctrl)
#pybind11#            self.assertFloatsAlmostEqual(field1.getCoefficients(), field2.getCoefficients())
#pybind11#            self.assertEqual(field1.getBBox(), field2.getBBox())
#pybind11#            config3 = lsst.afw.math.ChebyshevBoundedField.ConfigClass()
#pybind11#            config3.readControl(ctrl)
#pybind11#            if ctrl.orderX > 0:
#pybind11#                config3.orderX -= 1
#pybind11#            if ctrl.orderY > 0:
#pybind11#                config3.orderY -= 1
#pybind11#            field3 = field1.truncate(config3.makeControl())
#pybind11#            for i in range(config3.orderY + 1):
#pybind11#                for j in range(config3.orderX + 1):
#pybind11#                    if config3.triangular and i + j > max(config3.orderX, config3.orderY):
#pybind11#                        self.assertEqual(field3.getCoefficients()[i, j], 0.0)
#pybind11#                    else:
#pybind11#                        self.assertEqual(field3.getCoefficients()[i, j], field1.getCoefficients()[i, j])
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
