#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from builtins import next
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
#pybind11#import itertools
#pybind11#import math
#pybind11#import unittest
#pybind11#
#pybind11#import numpy as np
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.pex.exceptions as pexExceptions
#pybind11#
#pybind11#
#pybind11#def nrange(num, start, delta):
#pybind11#    """Return an array of num floats starting with start and incrementing by delta
#pybind11#    """
#pybind11#    return np.arange(start, start + (delta * (num - 0.1)), delta)
#pybind11#
#pybind11#
#pybind11#def sinc(x):
#pybind11#    """Return the normalized sinc function: sinc(x) = sin(pi * x) / (pi * x)
#pybind11#    """
#pybind11#    if abs(x) < 1.0e-15:
#pybind11#        return 1.0
#pybind11#    return math.sin(math.pi * x) / (math.pi * x)
#pybind11#
#pybind11#
#pybind11#def referenceChebyshev1(x, n):
#pybind11#    """Reference implementation of Chebyshev polynomials of the first kind
#pybind11#
#pybind11#    f(x) = T_n(x)
#pybind11#    """
#pybind11#    # from Wikipedia
#pybind11#    if n == 0:
#pybind11#        return 1.0
#pybind11#    if n == 1:
#pybind11#        return x
#pybind11#    return (2.0 * x * referenceChebyshev1(x, n-1)) - referenceChebyshev1(x, n-2)
#pybind11#
#pybind11#
#pybind11#def referenceChebyshev1Polynomial1(x, params):
#pybind11#    """Reference implementation of a 1-D polynomial of Chebyshev polynomials of the first kind
#pybind11#
#pybind11#    f(x) = params[0] T_0(x) + params[1] T_1(x) + params[2] T_2(x)
#pybind11#    """
#pybind11#    retVal = 0.0
#pybind11#    for ii in range(len(params)-1, -1, -1):
#pybind11#        retVal += params[ii] * referenceChebyshev1(x, ii)
#pybind11#    return retVal
#pybind11#
#pybind11#
#pybind11#def referenceChebyshev1Polynomial2(x, y, params):
#pybind11#    """Reference implementation of a 2-D polynomial of Chebyshev polynomials of the first kind
#pybind11#
#pybind11#    f(x) =   params[0] T_0(x) T_0(y)                                                        # order 0
#pybind11#           + params[1] T_1(x) T_0(y) + params[2] T_0(x) T_1(y)                              # order 1
#pybind11#           + params[3] T_2(x) T_0(y) + params[4] T_1(x) T_1(y) + params[5] T_0(x) T_2(y)    # order 2
#pybind11#           + ...
#pybind11#
#pybind11#    Raise RuntimeError if the number of parameters does not match an integer order.
#pybind11#    """
#pybind11#    retVal = 0.0
#pybind11#    order = 0
#pybind11#    y_order = 0
#pybind11#    for ii in range(0, len(params)):
#pybind11#        x_order = order - y_order
#pybind11#        retVal += params[ii] * referenceChebyshev1(x, x_order) * referenceChebyshev1(y, y_order)
#pybind11#        if x_order > 0:
#pybind11#            y_order += 1
#pybind11#        else:
#pybind11#            order += 1
#pybind11#            y_order = 0
#pybind11#    if y_order != 0:
#pybind11#        raise RuntimeError("invalid # of parameters=%d" % (len(params),))
#pybind11#    return retVal
#pybind11#
#pybind11#
#pybind11#class FunctionTestCase(lsst.utils.tests.TestCase):
#pybind11#    def setUp(self):
#pybind11#        self.normErr = "Invalid {0} normalization: min={1}, max={2}, min/max norm=({3}, {4}) != (-1, 1)"
#pybind11#        # We need a slightly larger than the full floating point tolerance for many of these tests.
#pybind11#        self.atol = 5e-14
#pybind11#
#pybind11#    def testChebyshev1Function1D(self):
#pybind11#        errMsg = "{}: {} != {} for x={}, xMin={}, xMax={}, xNorm={}, params={}; {}"
#pybind11#        maxOrder = 6
#pybind11#        deltaParam = 0.3
#pybind11#        ranges = ((-1, 1), (-1, 0), (0, 1), (-17, -2), (-65.3, 2.132))
#pybind11#        rangeIter = itertools.cycle(ranges)
#pybind11#        nPoints = 9
#pybind11#
#pybind11#        for order in range(maxOrder + 1):
#pybind11#            xMin, xMax = next(rangeIter)
#pybind11#            xMean = (xMin + xMax) / 2.0
#pybind11#            xDelta = (xMax - xMin) / float(nPoints - 1)
#pybind11#
#pybind11#            f = afwMath.Chebyshev1Function1D(order, xMin, xMax)
#pybind11#            numParams = f.getNParameters()
#pybind11#            params = np.arange(deltaParam, deltaParam * numParams + (deltaParam / 2.0), deltaParam)
#pybind11#            f.setParameters(params)
#pybind11#            g = afwMath.Chebyshev1Function1D(params, xMin, xMax)
#pybind11#            h = f.clone()
#pybind11#
#pybind11#            self.assertEqual(f.getNParameters(), g.getNParameters())
#pybind11#
#pybind11#            self.assertEqual(f.getMinX(), xMin)
#pybind11#            self.assertEqual(f.getMaxX(), xMax)
#pybind11#            self.assertEqual(f.getOrder(), order)
#pybind11#
#pybind11#            self.assertEqual(g.getMinX(), xMin)
#pybind11#            self.assertEqual(g.getMaxX(), xMax)
#pybind11#            self.assertEqual(g.getOrder(), order)
#pybind11#
#pybind11#            minXNorm = None
#pybind11#            maxXNorm = None
#pybind11#            for x in np.arange(xMin, xMax + xDelta/2.0, xDelta):
#pybind11#                xNorm = 2.0 * (x - xMean) / float(xMax - xMin)
#pybind11#                if minXNorm is None or xNorm < minXNorm:
#pybind11#                    minXNorm = xNorm
#pybind11#                if maxXNorm is None or xNorm > maxXNorm:
#pybind11#                    maxXNorm = xNorm
#pybind11#
#pybind11#                predVal = referenceChebyshev1Polynomial1(xNorm, params)
#pybind11#                msg = errMsg.format(type(f).__name__, f(x), predVal, x, xMin, xMax, xNorm, params,
#pybind11#                                    "order constructor")
#pybind11#                self.assertFloatsAlmostEqual(f(x), predVal, msg=msg, atol=self.atol)
#pybind11#                msg = errMsg.format(type(g).__name__, g(x), predVal, x, xMin, xMax, xNorm, params,
#pybind11#                                    "params constructor")
#pybind11#                self.assertFloatsAlmostEqual(g(x), predVal, msg=msg, atol=self.atol)
#pybind11#                msg = errMsg.format(type(h).__name__, h(x), predVal, x, xMin, xMax, xNorm, params, "clone")
#pybind11#                self.assertFloatsAlmostEqual(h(x), predVal, msg=msg, atol=self.atol)
#pybind11#
#pybind11#            msg = self.normErr.format("x", xMin, xMax, minXNorm, maxXNorm)
#pybind11#            self.assertFloatsAlmostEqual(minXNorm, -1., msg=msg, atol=self.atol)
#pybind11#            self.assertFloatsAlmostEqual(maxXNorm, 1., msg=msg, atol=self.atol)
#pybind11#
#pybind11#    def testChebyshev1Function2D(self):
#pybind11#        errMsg = ("{}: {} != {} for x={}, xMin={}, xMax={}, xNorm={}, "
#pybind11#                  "yMin={}, yMax={}, yNorm={}, params={}; {}")
#pybind11#        maxOrder = 6
#pybind11#        deltaParam = 0.3
#pybind11#        ranges = ((-1, 1), (-1, 0), (0, 1), (-17, -2), (-65.3, 2.132))
#pybind11#        xRangeIter = itertools.cycle(ranges)
#pybind11#        yRangeIter = itertools.cycle(ranges)
#pybind11#        next(yRangeIter)  # make x and y ranges off from each other
#pybind11#        nPoints = 7  # number of points in x and y at which to test the functions
#pybind11#
#pybind11#        for order in range(maxOrder + 1):
#pybind11#            xMin, xMax = next(xRangeIter)
#pybind11#            xMean = (xMin + xMax) / 2.0
#pybind11#            xDelta = (xMax - xMin) / float(nPoints - 1)
#pybind11#
#pybind11#            yMin, yMax = next(yRangeIter)
#pybind11#            yMean = (yMin + yMax) / 2.0
#pybind11#            yDelta = (yMax - yMin) / float(nPoints - 1)
#pybind11#
#pybind11#            xyRange = afwGeom.Box2D(afwGeom.Point2D(xMin, yMin), afwGeom.Point2D(xMax, yMax))
#pybind11#
#pybind11#            f = afwMath.Chebyshev1Function2D(order, xyRange)
#pybind11#            numParams = f.getNParameters()
#pybind11#            params = nrange(numParams, deltaParam, deltaParam)
#pybind11#            f.setParameters(params)
#pybind11#            g = afwMath.Chebyshev1Function2D(params, xyRange)
#pybind11#            h = f.clone()
#pybind11#
#pybind11#            self.assertEqual(f.getNParameters(), g.getNParameters())
#pybind11#            self.assertEqual(f.getNParameters(), h.getNParameters())
#pybind11#
#pybind11#            self.assertEqual(f.getXYRange(), xyRange)
#pybind11#            self.assertEqual(f.getOrder(), order)
#pybind11#
#pybind11#            self.assertEqual(g.getXYRange(), xyRange)
#pybind11#            self.assertEqual(g.getOrder(), order)
#pybind11#
#pybind11#            # vary x in the inner loop to exercise the caching
#pybind11#            minYNorm = None
#pybind11#            maxYNorm = None
#pybind11#            for y in np.arange(yMin, yMax + yDelta/2.0, yDelta):
#pybind11#                yNorm = 2.0 * (y - yMean) / float(yMax - yMin)
#pybind11#                if minYNorm is None or yNorm < minYNorm:
#pybind11#                    minYNorm = yNorm
#pybind11#                if maxYNorm is None or yNorm > maxYNorm:
#pybind11#                    maxYNorm = yNorm
#pybind11#
#pybind11#                minXNorm = None
#pybind11#                maxXNorm = None
#pybind11#                for x in np.arange(xMin, xMax + xDelta/2.0, xDelta):
#pybind11#                    xNorm = 2.0 * (x - xMean) / float(xMax - xMin)
#pybind11#                    if minXNorm is None or xNorm < minXNorm:
#pybind11#                        minXNorm = xNorm
#pybind11#                    if maxXNorm is None or xNorm > maxXNorm:
#pybind11#                        maxXNorm = xNorm
#pybind11#
#pybind11#                        predVal = referenceChebyshev1Polynomial2(xNorm, yNorm, params)
#pybind11#
#pybind11#                        msg = errMsg.format(type(f).__name__, f(x, y), predVal, x, xMin, xMax, xNorm,
#pybind11#                                            yMin, yMax, yNorm, params, "order constructor")
#pybind11#                        self.assertFloatsAlmostEqual(f(x, y), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#                        msg = errMsg.format(type(g).__name__, g(x, y), predVal, x, xMin, xMax, xNorm,
#pybind11#                                            yMin, yMax, yNorm, params, "params constructor")
#pybind11#                        self.assertFloatsAlmostEqual(g(x, y), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#                        msg = errMsg.format(type(h).__name__, h(x, y), predVal, x, xMin, xMax, xNorm,
#pybind11#                                            yMin, yMax, yNorm, params, "order")
#pybind11#                        self.assertFloatsAlmostEqual(h(x, y), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#
#pybind11#                msg = self.normErr.format("x", xMin, xMax, minXNorm, maxXNorm)
#pybind11#                self.assertFloatsAlmostEqual(minXNorm, -1., msg=msg, atol=self.atol)
#pybind11#                self.assertFloatsAlmostEqual(maxXNorm, 1., msg=msg, atol=self.atol)
#pybind11#
#pybind11#            msg = self.normErr.format("y", yMin, yMax, minYNorm, maxYNorm)
#pybind11#            self.assertFloatsAlmostEqual(minYNorm, -1., msg=msg, atol=self.atol)
#pybind11#            self.assertFloatsAlmostEqual(maxYNorm, 1., msg=msg, atol=self.atol)
#pybind11#
#pybind11#        # test that the number of parameters is correct for the given order
#pybind11#        def numParamsFromOrder(order):
#pybind11#            return (order + 1) * (order + 2) // 2
#pybind11#        MaxOrder = 13
#pybind11#        for order in range(MaxOrder+1):
#pybind11#            f = afwMath.Chebyshev1Function2D(order)
#pybind11#            predNParams = numParamsFromOrder(order)
#pybind11#            self.assertEqual(f.getNParameters(), predNParams)
#pybind11#            afwMath.Chebyshev1Function2D(np.zeros(predNParams, dtype=float))
#pybind11#
#pybind11#        # test that the wrong number of parameters raises an exception
#pybind11#        validNumParams = set()
#pybind11#        for order in range(MaxOrder+1):
#pybind11#            validNumParams.add(numParamsFromOrder(order))
#pybind11#        for numParams in range(numParamsFromOrder(MaxOrder)):
#pybind11#            if numParams in validNumParams:
#pybind11#                continue
#pybind11#            with self.assertRaises(pexExceptions.InvalidParameterError):
#pybind11#                afwMath.Chebyshev1Function2D(np.zeros(numParams, dtype=float))
#pybind11#
#pybind11#        # test that changing parameters clears the cache
#pybind11#        # for simplicity use the xyRange that requires no normalization
#pybind11#        order = 3
#pybind11#        numParams = numParamsFromOrder(order)
#pybind11#        f = afwMath.Chebyshev1Function2D(order)
#pybind11#        xyRange = afwGeom.Box2D(afwGeom.Point2D(-1.0, -1.0), afwGeom.Point2D(1.0, 1.0))
#pybind11#        x = 0.5
#pybind11#        y = -0.24
#pybind11#        for addValue in (0.0, 0.2):
#pybind11#            params = nrange(numParams, deltaParam + addValue, deltaParam)
#pybind11#            f.setParameters(params)
#pybind11#            predVal = referenceChebyshev1Polynomial2(x, y, params)
#pybind11#            msg = "%s != %s for x=%s, y=%s, params=%s" % (f(x, y), predVal, x, y, params)
#pybind11#            self.assertFloatsAlmostEqual(f(x, y), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#
#pybind11#    def testChebyshev1Function2DTruncate(self):
#pybind11#        errMsg = ("{} != {} = {} for x={}, xMin={}, xMax={}, xNorm={},"
#pybind11#                  " yMin={}, yMax={}, yNorm={}, truncParams={}; order constructor")
#pybind11#
#pybind11#        maxOrder = 6
#pybind11#        deltaParam = 0.3
#pybind11#        ranges = ((-1, 1), (-17, -2), (-65.3, 2.132))
#pybind11#        xRangeIter = itertools.cycle(ranges)
#pybind11#        yRangeIter = itertools.cycle(ranges)
#pybind11#        next(yRangeIter)  # make x and y ranges off from each other
#pybind11#        nPoints = 7  # number of points in x and y at which to test the functions
#pybind11#
#pybind11#        for order in range(maxOrder + 1):
#pybind11#            xMin, xMax = next(xRangeIter)
#pybind11#            xMean = (xMin + xMax) / 2.0
#pybind11#            xDelta = (xMax - xMin) / float(nPoints - 1)
#pybind11#
#pybind11#            yMin, yMax = next(yRangeIter)
#pybind11#            yMean = (yMin + yMax) / 2.0
#pybind11#            yDelta = (yMax - yMin) / float(nPoints - 1)
#pybind11#
#pybind11#            xyRange = afwGeom.Box2D(afwGeom.Point2D(xMin, yMin), afwGeom.Point2D(xMax, yMax))
#pybind11#
#pybind11#            fullNParams = afwMath.Chebyshev1Function2D.nParametersFromOrder(order)
#pybind11#            fullParams = nrange(fullNParams, deltaParam, deltaParam)
#pybind11#            fullPoly = afwMath.Chebyshev1Function2D(fullParams, xyRange)
#pybind11#
#pybind11#            for tooBigTruncOrder in range(order + 1, order + 3):
#pybind11#                with self.assertRaises(pexExceptions.InvalidParameterError):
#pybind11#                    fullPoly.truncate(tooBigTruncOrder)
#pybind11#
#pybind11#            for truncOrder in range(order + 1):
#pybind11#                truncNParams = fullPoly.nParametersFromOrder(truncOrder)
#pybind11#                truncParams = fullParams[0:truncNParams]
#pybind11#
#pybind11#                f = fullPoly.truncate(truncOrder)
#pybind11#                self.assertEqual(f.getNParameters(), truncNParams)
#pybind11#
#pybind11#                g = afwMath.Chebyshev1Function2D(fullParams[0:truncNParams], xyRange)
#pybind11#
#pybind11#                self.assertEqual(f.getNParameters(), g.getNParameters())
#pybind11#
#pybind11#                self.assertEqual(f.getOrder(), truncOrder)
#pybind11#                self.assertEqual(f.getXYRange(), xyRange)
#pybind11#
#pybind11#                self.assertEqual(g.getOrder(), truncOrder)
#pybind11#                self.assertEqual(g.getXYRange(), xyRange)
#pybind11#
#pybind11#                minXNorm = None
#pybind11#                maxXNorm = None
#pybind11#                for x in np.arange(xMin, xMax + xDelta/2.0, xDelta):
#pybind11#                    xNorm = 2.0 * (x - xMean) / float(xMax - xMin)
#pybind11#                    if minXNorm is None or xNorm < minXNorm:
#pybind11#                        minXNorm = xNorm
#pybind11#                    if maxXNorm is None or xNorm > maxXNorm:
#pybind11#                        maxXNorm = xNorm
#pybind11#
#pybind11#                    minYNorm = None
#pybind11#                    maxYNorm = None
#pybind11#                    for y in np.arange(yMin, yMax + yDelta/2.0, yDelta):
#pybind11#                        yNorm = 2.0 * (y - yMean) / float(yMax - yMin)
#pybind11#                        if minYNorm is None or yNorm < minYNorm:
#pybind11#                            minYNorm = yNorm
#pybind11#                        if maxYNorm is None or yNorm > maxYNorm:
#pybind11#                            maxYNorm = yNorm
#pybind11#
#pybind11#                            msg = errMsg.format(type(f).__name__, f(x, y), g(x, y), type(g).__name__,
#pybind11#                                                x, xMin, xMax, xNorm, yMin, yMax, yNorm, truncParams)
#pybind11#                            self.assertFloatsAlmostEqual(f(x, y), g(x, y), msg=msg)
#pybind11#
#pybind11#                    msg = self.normErr.format("y", yMin, yMax, minYNorm, maxYNorm)
#pybind11#                    self.assertFloatsAlmostEqual(minYNorm, -1.0, msg=msg, atol=self.atol, rtol=None)
#pybind11#                    self.assertFloatsAlmostEqual(maxYNorm, 1.0, msg=msg, atol=self.atol, rtol=None)
#pybind11#
#pybind11#                msg = self.normErr.format("x", xMin, xMax, minXNorm, maxXNorm)
#pybind11#                self.assertFloatsAlmostEqual(minXNorm, -1.0, msg=msg, atol=self.atol, rtol=None)
#pybind11#                self.assertFloatsAlmostEqual(maxXNorm, 1.0, msg=msg, atol=self.atol, rtol=None)
#pybind11#
#pybind11#    def testGaussianFunction1D(self):
#pybind11#        def basicGaussian(x, sigma):
#pybind11#            return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-x**2 / (2.0 * sigma**2))
#pybind11#
#pybind11#        f = afwMath.GaussianFunction1D(1.0)
#pybind11#        for xsigma in (0.1, 1.0, 3.0):
#pybind11#            f.setParameters((xsigma,))
#pybind11#            g = f.clone()
#pybind11#            xdelta = xsigma / 10.0
#pybind11#            fSum = 0.0
#pybind11#            for x in np.arange(-xsigma * 20, xsigma * 20.01, xdelta):
#pybind11#                predVal = basicGaussian(x, xsigma)
#pybind11#                fSum += predVal
#pybind11#                msg = "%s = %s != %s for x=%s, xsigma=%s" % (type(f).__name__, f(x), predVal, x, xsigma)
#pybind11#                self.assertFloatsAlmostEqual(f(x), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#                msg += "; clone"
#pybind11#                self.assertFloatsAlmostEqual(g(x), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#            approxArea = fSum * xdelta
#pybind11#            msg = "%s area = %s != 1.0 for xsigma=%s" % (type(f).__name__, approxArea, xsigma)
#pybind11#            self.assertFloatsAlmostEqual(approxArea, 1.0, msg=msg, atol=self.atol, rtol=None)
#pybind11#
#pybind11#    def testGaussianFunction2D(self):
#pybind11#        """Note: Assumes GaussianFunction1D is correct (tested elsewhere)."""
#pybind11#        errMsg = "{} = {} != {} for pos1={}, pos2={}, x={}, y={}, sigma1={}, sigma2={}, angle={}"
#pybind11#        areaMsg = "%s area = %s != 1.0 for sigma1=%s, sigma2=%s"
#pybind11#        f = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        f1 = afwMath.GaussianFunction1D(1.0)
#pybind11#        f2 = afwMath.GaussianFunction1D(1.0)
#pybind11#        for sigma1 in (0.1, 1.0, 3.0):
#pybind11#            for sigma2 in (0.1, 1.0, 3.0):
#pybind11#                for angle in (0.0, 0.4, 1.1):
#pybind11#                    sinNegAngle = math.sin(-angle)
#pybind11#                    cosNegAngle = math.cos(-angle)
#pybind11#                    f.setParameters((sigma1, sigma2, angle))
#pybind11#                    g = f.clone()
#pybind11#                    f1.setParameters((sigma1,))
#pybind11#                    f2.setParameters((sigma2,))
#pybind11#                    fSum = 0.0
#pybind11#                    delta1 = sigma1 / 5.0
#pybind11#                    delta2 = sigma2 / 5.0
#pybind11#                    for pos1 in np.arange(-sigma1 * 5, sigma1 * 5.01, delta1):
#pybind11#                        for pos2 in np.arange(-sigma2 * 5.0, sigma2 * 5.01, delta2):
#pybind11#                            x = (cosNegAngle * pos1) + (sinNegAngle * pos2)
#pybind11#                            y = (-sinNegAngle * pos1) + (cosNegAngle * pos2)
#pybind11#                            predVal = f1(pos1) * f2(pos2)
#pybind11#                            fSum += predVal
#pybind11#                            msg = errMsg.format(type(f).__name__, f(x, y), predVal,
#pybind11#                                                pos1, pos2, x, y, sigma1, sigma2, angle)
#pybind11#                            self.assertFloatsAlmostEqual(f(x, y), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#                            msg = errMsg.format(type(g).__name__, g(x, y), predVal,
#pybind11#                                                pos1, pos2, x, y, sigma1, sigma2, angle) + "; clone"
#pybind11#                            self.assertFloatsAlmostEqual(g(x, y), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#                    approxArea = fSum * delta1 * delta2
#pybind11#                    msg = areaMsg % (type(f).__name__, approxArea, sigma1, sigma2)
#pybind11#                    # approxArea is very approximate, so we need a high tolerance threshold.
#pybind11#                    self.assertFloatsAlmostEqual(approxArea, 1.0, msg=msg, atol=1e-6, rtol=None)
#pybind11#
#pybind11#    def testDoubleGaussianFunction2D(self):
#pybind11#        """Note: Assumes GaussianFunction2D is correct (tested elsewhere)."""
#pybind11#        errMsg = "{} = {} != {} for x={}, y={}, sigma1={}, sigma2={}, b={}"
#pybind11#        areaMsg = "{} area = {} != 1.0 for sigma1={}, sigma2={}"
#pybind11#        f = afwMath.DoubleGaussianFunction2D(1.0, 1.0)
#pybind11#        f1 = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        f2 = afwMath.GaussianFunction2D(1.0, 1.0, 0.0)
#pybind11#        for sigma1 in (1.0,):
#pybind11#            for sigma2 in (0.5, 2.0):
#pybind11#                for b in (0.0, 0.2, 2.0):
#pybind11#                    f.setParameters((sigma1, sigma2, b))
#pybind11#                    g = f.clone()
#pybind11#                    f1.setParameters((sigma1, sigma1, 0.0))
#pybind11#                    f2.setParameters((sigma2, sigma2, 0.0))
#pybind11#                    sigma1Sq = sigma1**2
#pybind11#                    sigma2Sq = sigma2**2
#pybind11#                    f1Mult = b * sigma2Sq / sigma1Sq
#pybind11#                    allMult = sigma1Sq / (sigma1Sq + (b * sigma2Sq))
#pybind11#                    fSum = 0.0
#pybind11#                    maxsigma = max(sigma1, sigma2)
#pybind11#                    minsigma = min(sigma1, sigma2)
#pybind11#                    delta = minsigma / 5.0
#pybind11#                    for y in np.arange(-maxsigma * 5, maxsigma * 5.01, delta):
#pybind11#                        for x in np.arange(-maxsigma * 5.0, maxsigma * 5.01, delta):
#pybind11#                            predVal = (f1(x, y) + (f1Mult * f2(x, y))) * allMult
#pybind11#                            fSum += predVal
#pybind11#                            msg = errMsg.format(type(f).__name__, f(x, y), predVal, x, y, sigma1, sigma2, b)
#pybind11#                            self.assertFloatsAlmostEqual(f(x, y), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#                            msg = errMsg.format(type(g).__name__, g(x, y), predVal,
#pybind11#                                                x, y, sigma1, sigma2, b) + "; clone"
#pybind11#                            self.assertFloatsAlmostEqual(g(x, y), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#                    approxArea = fSum * delta**2
#pybind11#                    msg = areaMsg.format(type(f).__name__, approxArea, sigma1, sigma2)
#pybind11#                    # approxArea is very approximate, so we need a high tolerance threshold.
#pybind11#                    self.assertFloatsAlmostEqual(approxArea, 1.0, msg=msg, atol=1e-6, rtol=None)
#pybind11#
#pybind11#    def testIntegerDeltaFunction2D(self):
#pybind11#        def basicDelta(x, xo):
#pybind11#            return (x == xo)
#pybind11#
#pybind11#        errMsg = "{} = {} != {} for x={}, y={}, xo={}, yo={}"
#pybind11#        for xo in np.arange(-5.0, 5.0, 1.0):
#pybind11#            for yo in np.arange(-5.0, 5.0, 1.0):
#pybind11#                f = afwMath.IntegerDeltaFunction2D(xo, yo)
#pybind11#                g = f.clone()
#pybind11#                for x in np.arange(-5.0, 5.0, 1.0):
#pybind11#                    for y in np.arange(-5.0, 5.0, 1.0):
#pybind11#                        predVal = basicDelta(x, xo) * basicDelta(y, yo)
#pybind11#                        msg = errMsg.format(type(f).__name__, f(x, y), predVal, x, y, xo, yo)
#pybind11#                        self.assertFloatsAlmostEqual(f(x, y), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#                        msg = errMsg.format(type(g).__name__, g(x, y), predVal, x, y, xo, yo) + "; clone"
#pybind11#                        self.assertFloatsAlmostEqual(g(x, y), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#
#pybind11#    def testLanczosFunction1D(self):
#pybind11#        def basicLanczos1(x, n):
#pybind11#            return sinc(x) * sinc(x / float(n))
#pybind11#
#pybind11#        errMsg = "{} = {} != {} for n={}, x={}, xOffset={}, xAdj={}"
#pybind11#        for n in range(1, 5):
#pybind11#            f = afwMath.LanczosFunction1D(n)
#pybind11#            self.assertEquals(f.getOrder(), n)
#pybind11#
#pybind11#            for xOffset in (-10.0, 0.0, 0.05):
#pybind11#                f.setParameters((xOffset,))
#pybind11#                g = f.clone()
#pybind11#                for x in np.arange(-10.0, 10.1, 0.50):
#pybind11#                    xAdj = x - xOffset
#pybind11#                    predVal = basicLanczos1(xAdj, n)
#pybind11#                    msg = errMsg.format(type(f).__name__, f(x), predVal, n, x, xOffset, xAdj)
#pybind11#                    self.assertFloatsAlmostEqual(f(x), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#                    msg = errMsg.format(type(g).__name__, g(x), predVal, n, x, xOffset, xAdj) + "; clone"
#pybind11#                    self.assertFloatsAlmostEqual(g(x), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#
#pybind11#    def testLanczosFunction2D(self):
#pybind11#        def basicLanczos1(x, n):
#pybind11#            return sinc(x) * sinc(x / float(n))
#pybind11#
#pybind11#        errMsg = "{} = {} != {} for n={}, x={}, xOffset={}, yOffset={}, xAdj={}, yAdj={}"
#pybind11#        for n in range(1, 5):
#pybind11#            f = afwMath.LanczosFunction2D(n)
#pybind11#            self.assertEquals(f.getOrder(), n)
#pybind11#
#pybind11#            for xOffset in (-10.0, 0.0, 0.05):
#pybind11#                for yOffset in (-0.01, 0.0, 7.5):
#pybind11#                    f.setParameters((xOffset, yOffset))
#pybind11#                    g = f.clone()
#pybind11#                    for x in np.arange(-10.0, 10.1, 2.0):
#pybind11#                        for y in np.arange(-10.0, 10.1, 2.0):
#pybind11#                            xAdj = x - xOffset
#pybind11#                            yAdj = y - yOffset
#pybind11#                            predVal = basicLanczos1(xAdj, n) * basicLanczos1(yAdj, n)
#pybind11#                            msg = errMsg.format(type(f).__name__, f(x, y), predVal, n,
#pybind11#                                                x, xOffset, yOffset, xAdj, yAdj)
#pybind11#                            self.assertFloatsAlmostEqual(f(x, y), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#                            msg = errMsg.format(type(g).__name__, g(x, y), predVal, n,
#pybind11#                                                x, xOffset, yOffset, xAdj, yAdj)
#pybind11#                            self.assertFloatsAlmostEqual(g(x, y), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#
#pybind11#    def testPolynomialFunction1D(self):
#pybind11#        def basic1DPoly(x, params):
#pybind11#            ii = len(params) - 1
#pybind11#            retVal = params[ii]
#pybind11#            while ii > 0:
#pybind11#                ii -= 1
#pybind11#                retVal = retVal * x + params[ii]
#pybind11#            return retVal
#pybind11#
#pybind11#        maxOrder = 4
#pybind11#        deltaParam = 0.3
#pybind11#        errMsg = "{} = {} != {} for x={}, params={}; {}"
#pybind11#
#pybind11#        # test value using order constructor
#pybind11#        for order in range(maxOrder):
#pybind11#            numParams = order + 1
#pybind11#            params = nrange(numParams, deltaParam, deltaParam)
#pybind11#            f = afwMath.PolynomialFunction1D(params)
#pybind11#            g = afwMath.PolynomialFunction1D(order)
#pybind11#            g.setParameters(params)
#pybind11#            h = f.clone()
#pybind11#
#pybind11#            self.assertEqual(f.getOrder(), order)
#pybind11#            self.assertEqual(g.getOrder(), order)
#pybind11#
#pybind11#            for x in np.arange(-10.0, 10.1, 1.0):
#pybind11#                predVal = basic1DPoly(x, params)
#pybind11#                msg = errMsg.format(type(f).__name__, f(x), predVal, x, params, "params constructor")
#pybind11#                self.assertFloatsAlmostEqual(f(x), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#                msg = errMsg.format(type(g).__name__, g(x), predVal, x, params, "order constructor")
#pybind11#                self.assertFloatsAlmostEqual(g(x), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#                msg = errMsg.format(type(h).__name__, h(x), predVal, x, params, "clone")
#pybind11#                self.assertFloatsAlmostEqual(h(x), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#
#pybind11#    def testPolynomialFunction2D(self):
#pybind11#        def basic2DPoly(x, y, params):
#pybind11#            retVal = 0
#pybind11#            numParams = len(params)
#pybind11#            order = 0
#pybind11#            ii = 0
#pybind11#            while True:
#pybind11#                for yOrder in range(order+1):
#pybind11#                    xOrder = order - yOrder
#pybind11#                    retVal += params[ii] * x**xOrder * y**yOrder
#pybind11#                    ii += 1
#pybind11#                    if ii >= numParams:
#pybind11#                        if xOrder != 0:
#pybind11#                            raise RuntimeError("invalid # params=%d" % (numParams,))
#pybind11#                        return retVal
#pybind11#                order += 1
#pybind11#
#pybind11#        numParamsList = (1, 3, 6, 10)
#pybind11#        deltaParam = 0.3
#pybind11#        errMsg = "{} = {} != {} for x={}, y={}, params={}; {}"
#pybind11#
#pybind11#        # test function values
#pybind11#        for order, numParams in enumerate(numParamsList):
#pybind11#            params = nrange(numParams, deltaParam, deltaParam)
#pybind11#            f = afwMath.PolynomialFunction2D(params)
#pybind11#            g = afwMath.PolynomialFunction2D(order)
#pybind11#            g.setParameters(params)
#pybind11#            h = f.clone()
#pybind11#
#pybind11#            self.assertEqual(f.getOrder(), order)
#pybind11#            self.assertEqual(g.getOrder(), order)
#pybind11#
#pybind11#            # vary x in the inner loop to exercise the caching
#pybind11#            for y in np.arange(-10.0, 10.1, 2.5):
#pybind11#                for x in np.arange(-10.0, 10.1, 2.5):
#pybind11#                    predVal = basic2DPoly(x, y, params)
#pybind11#                    msg = errMsg.format(type(f).__name__, f(x, y), predVal,
#pybind11#                                        x, y, params, "params constructor")
#pybind11#                    self.assertFloatsAlmostEqual(f(x, y), predVal, msg=msg, atol=2e-12, rtol=None)
#pybind11#                    msg = errMsg.format(type(g).__name__, g(x, y), predVal, x, y, params, "order constructor")
#pybind11#                    self.assertFloatsAlmostEqual(g(x, y), predVal, msg=msg, atol=2e-12, rtol=None)
#pybind11#                    msg = errMsg.format(type(h).__name__, h(x, y), predVal, x, y, params, "clone")
#pybind11#                    self.assertFloatsAlmostEqual(h(x, y), predVal, msg=msg, atol=2e-12, rtol=None)
#pybind11#
#pybind11#        # test that the number of parameters is correct for the given order
#pybind11#        def numParamsFromOrder(order):
#pybind11#            return (order + 1) * (order + 2) // 2
#pybind11#        MaxOrder = 13
#pybind11#        for order in range(MaxOrder+1):
#pybind11#            f = afwMath.PolynomialFunction2D(order)
#pybind11#            predNParams = numParamsFromOrder(order)
#pybind11#            self.assertEqual(f.getNParameters(), predNParams)
#pybind11#            afwMath.PolynomialFunction2D(np.zeros(predNParams, dtype=float))
#pybind11#
#pybind11#        # test that the wrong number of parameters raises an exception
#pybind11#        validNumParams = set()
#pybind11#        for order in range(MaxOrder+1):
#pybind11#            validNumParams.add(numParamsFromOrder(order))
#pybind11#        for numParams in range(numParamsFromOrder(MaxOrder)):
#pybind11#            if numParams in validNumParams:
#pybind11#                continue
#pybind11#            with self.assertRaises(pexExceptions.InvalidParameterError):
#pybind11#                afwMath.PolynomialFunction2D(np.zeros(numParams, dtype=float))
#pybind11#
#pybind11#        # test that changing parameters clears the cache
#pybind11#        order = 3
#pybind11#        numParams = numParamsFromOrder(order)
#pybind11#        f = afwMath.PolynomialFunction2D(order)
#pybind11#        x = 0.5
#pybind11#        y = -0.24
#pybind11#        for addValue in (0.0, 0.2):
#pybind11#            params = nrange(numParams, deltaParam + addValue, deltaParam)
#pybind11#            f.setParameters(params)
#pybind11#            predVal = basic2DPoly(x, y, params)
#pybind11#            msg = errMsg.format(type(f).__name__, f(x, y), predVal, x, y, params, "")
#pybind11#            self.assertFloatsAlmostEqual(f(x, y), predVal, msg=msg, atol=self.atol, rtol=None)
#pybind11#
#pybind11#    def testDFuncDParameters(self):
#pybind11#        """Test that we can differentiate the Function2 with respect to its parameters"""
#pybind11#        nOrder = 3
#pybind11#        params = []
#pybind11#        for i in range((nOrder + 1)*(nOrder + 2)//2):
#pybind11#            params.append(math.sin(1 + i))  # deterministic pretty-random numbers
#pybind11#
#pybind11#        f = afwMath.PolynomialFunction2D(params)
#pybind11#
#pybind11#        for (x, y) in [(2, 1), (1, 2), (2, 2)]:
#pybind11#            dFdC = f.getDFuncDParameters(x, y)
#pybind11#
#pybind11#            self.assertAlmostEqual(f(x, y), sum([params[i]*dFdC[i] for i in range(len(params))]))
#pybind11#
#pybind11#    def testCast(self):
#pybind11#        for instance in (afwMath.Chebyshev1Function1F(2), afwMath.GaussianFunction1F(1.0),
#pybind11#                         afwMath.LanczosFunction1F(3), afwMath.NullFunction1F(),
#pybind11#                         afwMath.PolynomialFunction1F(2)):
#pybind11#            Class = type(instance)
#pybind11#            base = instance.clone()
#pybind11#            self.assertEqual(type(base), afwMath.Function1F)
#pybind11#            derived = Class.cast(base)
#pybind11#            self.assertEqual(type(derived), Class)
#pybind11#        for instance in (afwMath.Chebyshev1Function1D(2), afwMath.GaussianFunction1D(1.0),
#pybind11#                         afwMath.LanczosFunction1D(3), afwMath.NullFunction1D(),
#pybind11#                         afwMath.PolynomialFunction1D(2)):
#pybind11#            Class = type(instance)
#pybind11#            base = instance.clone()
#pybind11#            self.assertEqual(type(base), afwMath.Function1D)
#pybind11#            derived = Class.cast(base)
#pybind11#            self.assertEqual(type(derived), Class)
#pybind11#        for instance in (afwMath.Chebyshev1Function2F(2), afwMath.GaussianFunction2F(1.0, 1.0),
#pybind11#                         afwMath.DoubleGaussianFunction2F(1.0),
#pybind11#                         afwMath.LanczosFunction2F(3), afwMath.NullFunction2F(),
#pybind11#                         afwMath.PolynomialFunction2F(2)):
#pybind11#            Class = type(instance)
#pybind11#            base = instance.clone()
#pybind11#            self.assertEqual(type(base), afwMath.Function2F)
#pybind11#            derived = Class.cast(base)
#pybind11#            self.assertEqual(type(derived), Class)
#pybind11#        for instance in (afwMath.Chebyshev1Function2D(2), afwMath.GaussianFunction2D(1.0, 1.0),
#pybind11#                         afwMath.DoubleGaussianFunction2D(1.0),
#pybind11#                         afwMath.LanczosFunction2D(3), afwMath.NullFunction2D(),
#pybind11#                         afwMath.PolynomialFunction2D(2)):
#pybind11#            Class = type(instance)
#pybind11#            base = instance.clone()
#pybind11#            self.assertEqual(type(base), afwMath.Function2D)
#pybind11#            derived = Class.cast(base)
#pybind11#            self.assertEqual(type(derived), Class)
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
