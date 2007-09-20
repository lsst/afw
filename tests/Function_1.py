import itertools
import math
import pdb                          # we may want to say pdb.set_trace()
import unittest

import numpy

import lsst.fw.Core.fwLib as fw
import lsst.mwi.tests as tests
import lsst.mwi.utils as mwiu

try:
    type(verbose)
except NameError:
    verbose = 0
    mwiu.Trace_setVerbosity("fw.kernel", verbose)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def sincpi(x):
    if abs(x) < 1.0e-15:
        return 1.0
    return math.sin(math.pi * x) / (math.pi * x)

class FunctionTestCase(unittest.TestCase):
    def testChebyshev1Function1D(self):
        """A test for Chebyshev1Function1D"""
        def basicCheby(x, order):
            # from Wikipedia
            if order == 0:
                return   1.0
            if order == 1:
                return   1.0 * x
            if order == 2:
                return   2.0 * x**2 -   1.0
            if order == 3:
                return   4.0 * x**3 -   3.0 * x
            if order == 4: 
                return   8.0 * x**4 -   8.0 * x**2 +   1.0
            if order == 5:
                return  16.0 * x**5 -  20.0 * x**3 +   5.0 * x
            if order == 6:
                return  32.0 * x**6 -  48.0 * x**4 +  18.0 * x**2 -   1.0
            if order == 7:
                return  64.0 * x**7 - 112.0 * x**5 +  56.0 * x**3 -   7.0 * x
            if order == 8:
                return 128.0 * x**8 - 256.0 * x**6 + 160.0 * x**4 -  32.0 * x**2 + 1.0
            if order == 9:
                return 256.0 * x**9 - 576.0 * x**7 + 432.0 * x**5 - 120.0 * x**3 + 9.0 * x
            raise ValueError("order %d > 9" % (order,))
        
        def basicChebyPoly(x, params):
            retVal = 0.0
            for ii in range(len(params)-1, -1, -1):
                retVal += params[ii] * basicCheby(x, ii)
            return retVal
        
        maxOrder = 9
        deltaCoeff = 0.3
        allCoeffs = numpy.arange(deltaCoeff, deltaCoeff * (maxOrder + 1) + (deltaCoeff / 2.0), deltaCoeff)
        ranges = ((-1, 1), (-1, 0), (0, 1), (-17, -2), (-65.3, 2.132))
        rangeIter = itertools.cycle(ranges)
        nPoints = 10
        
        for order in range(maxOrder + 1):
            minXNorm = None
            maxXNorm = None
            coeffs = allCoeffs[0: order + 1]
            xMin, xMax = rangeIter.next()
            xMean = (xMin + xMax) / 2.0
            xDelta = (xMax - xMin) / float(nPoints - 1)
            f = fw.Chebyshev1Function1D(coeffs, xMin, xMax)
            for x in numpy.arange(xMin, xMax + xDelta/2.0, xDelta):
                xNorm = 2.0 * (x - xMean) / float(xMax - xMin)
                if minXNorm == None or xNorm < minXNorm:
                    minXNorm = xNorm
                if maxXNorm == None or xNorm > maxXNorm:
                    maxXNorm = xNorm
                predVal = basicChebyPoly(xNorm, coeffs)
                if not numpy.allclose(predVal, f(x)):
                    self.fail("%s = %s != %s for x=%s, xMin=%s, xMax=%s, xNorm=%s, coeffs=%s" % \
                        (f.__class__.__name__, f(x), predVal, x, xMin, xMax, xNorm, coeffs))
            if not numpy.allclose((minXNorm, maxXNorm), (-1.0, 1.0)):
                raise RuntimeError("Invalid x normalization: xMin=%s, xMax=%s, min/max xNorm=(%s, %s) != (-1, 1)" %
                    (xMin, xMax, minXNorm, maxXNorm))
        
    def testGaussianFunction1D(self):
        """A test for GaussianFunction1D"""
        def basicGaussian(x, sigma):
            return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-x**2 / (2.0 * sigma**2))
        
        f = fw.GaussianFunction1D(1.0)
        for xsigma in (0.1, 1.0, 3.0):
            f.setParameters((xsigma,))
            for x in numpy.arange(-10.0, 10.1, 0.25):
                predVal = basicGaussian(x, xsigma)
                if not numpy.allclose(predVal, f(x)):
                    self.fail("%s = %s != %s for x=%s, xsigma=%s" % \
                        (f.__class__.__name__, f(x), predVal, x, xsigma))

    def testGaussianFunction2D(self):
        """A test for GaussianFunction2D
        Assumes GaussianFunction1D is correct (test it elsewhere)
        """
        f = fw.GaussianFunction2D(1.0, 1.0)
        fx = fw.GaussianFunction1D(1.0)
        fy = fw.GaussianFunction1D(1.0)
        for xsigma in (0.1, 1.0, 3.0):
            for ysigma in (0.1, 1.0, 3.0):
                f.setParameters((xsigma, ysigma))
                fx.setParameters((xsigma,))
                fy.setParameters((ysigma,))
                for y in numpy.arange(-2.5, 2.55, 0.5):
                    for x in numpy.arange(-10.0, 10.1, 2.0):
                        predVal = fx(x) * fy(y)
                        if not numpy.allclose(predVal, f(x,y)):
                            self.fail("%s = %s != %s for x=%s, y=%s, xsigma=%s, ysigma=%s" % \
                                (f.__class__.__name__, f(x,y), predVal, x, y, xsigma, ysigma))
    
    def testIntegerDeltaFunction2D(self):
        """A test for IntegerDeltaFunction2D"""
        def basicDelta(x, xo):
            return (x == xo)
        
        for xo in numpy.arange(-5.0, 5.0, 1.0):
            for yo in numpy.arange(-5.0, 5.0, 1.0):
                f = fw.IntegerDeltaFunction2D(xo, yo)
                for x in numpy.arange(-5.0, 5.0, 1.0):
                    for y in numpy.arange(-5.0, 5.0, 1.0):
                        predVal = basicDelta(x, xo) * basicDelta(y, yo)
                        if predVal != f(x,y):
                            self.fail("%s = %s != %s for x=%s, y=%s, xo=%s, yo=%s" % \
                                (f.__class__.__name__, f(x,y), predVal, x, y, xo, yo))
    
    def testLanczosFunction1D(self):
        """A test for LanczosFunction1D"""
        def basicLanczos1(x, n):
            return sincpi(x) * sincpi(x / float(n))

        for n in range(1, 5):
            f = fw.LanczosFunction1D(n)
            for xOffset in (-10.0, 0.0, 0.05):
                f.setParameters((xOffset,))
                for x in numpy.arange(-10.0, 10.1, 0.50):
                    xAdj = x - xOffset
                    predVal = basicLanczos1(xAdj, n)
                    if not numpy.allclose(predVal, f(x)):
                        self.fail("%s = %s != %s for n=%s, x=%s, xOffset=%s, xAdj=%s" % \
                            (f.__class__.__name__, f(x), predVal, n, x, xOffset, xAdj))

    def testLanczosFunction2D(self):
        """A test for LanczosFunction2D, the radial version"""
        def basicLanczos1(x, n):
            return sincpi(x) * sincpi(x / float(n))

        for n in range(1, 5):
            f = fw.LanczosFunction2D(n)
            for xOffset in (-10.0, 0.0, 0.05):
                for yOffset in (-0.01, 0.0, 7.5):
                    f.setParameters((xOffset, yOffset))
                    for x in numpy.arange(-10.0, 10.1, 2.0):
                        for y in numpy.arange(-10.0, 10.1, 2.0):
                            rad = math.sqrt((x - xOffset)**2 + (y - yOffset)**2)
                            predVal = basicLanczos1(rad, n)
                            if not numpy.allclose(predVal, f(x, y)):
                                self.fail("%s = %s != %s for n=%s, x=%s, xOffset=%s, yOffset=%s, rad=%s" % \
                                    (f.__class__.__name__, f(x,y), predVal, n, x, xOffset, yOffset, rad))
       
    def testLanczosSeparableFunction2D(self):
        """A test for LanczosSeparableFunction2D, the separable version"""
        def basicLanczos1(x, n):
            return sincpi(x) * sincpi(x / float(n))

        for n in range(1, 5):
            f = fw.LanczosSeparableFunction2D(n)
            for xOffset in (-10.0, 0.0, 0.05):
                for yOffset in (-0.01, 0.0, 7.5):
                    f.setParameters((xOffset, yOffset))
                    for x in numpy.arange(-10.0, 10.1, 2.0):
                        for y in numpy.arange(-10.0, 10.1, 2.0):
                            xAdj = x - xOffset
                            yAdj = y - yOffset
                            predVal = basicLanczos1(xAdj, n) * basicLanczos1(yAdj, n)
                            if not numpy.allclose(predVal, f(x, y)):
                                self.fail("%s = %s != %s for n=%s, x=%s, xOffset=%s, yOffset=%s, xAdj=%s, yAdj=%s" % \
                                    (f.__class__.__name__, f(x,y), predVal, n, x, xOffset, yOffset, xAdj, yAdj))
       
    def testPolynomialFunction1D(self):
        """A test for PolynomialFunction1D
        """
        def basic1DPoly(x, params):
            """1-dimensional polynomial function"""
            ii = len(params) - 1
            retVal = params[ii]
            while ii > 0:
                ii -= 1
                retVal = retVal * x + params[ii]
            return retVal
        
        maxOrder = 4
        deltaCoeff = 0.3
        allCoeffs = numpy.arange(deltaCoeff, deltaCoeff * (maxOrder + 1) + (deltaCoeff / 2.0), deltaCoeff)
        
        for order in range(maxOrder):
            coeffs = allCoeffs[0: order + 1]
            f = fw.PolynomialFunction1D(coeffs)
            for x in numpy.arange(-10.0, 10.1, 1.0):
                predVal = basic1DPoly(x, coeffs)
                if not numpy.allclose(predVal, f(x)):
                    self.fail("%s = %s != %s for x=%s, coeffs=%s" % \
                        (f.__class__.__name__, f(x), predVal, x, coeffs))

    def testPolynomialFunction2D(self):
        """A test for PolynomialFunction2D
        """
        def basic2DPoly(x, y, params):
            """2-dimensional polynomial function"""
            retVal = 0
            nParams = len(params)
            order = 0
            ii = 0
            while True:
                for yOrder in range(order+1):
                    xOrder = order - yOrder
                    retVal += params[ii] * x**xOrder * y**yOrder
                    ii += 1
                    if ii >= nParams:
                        if xOrder != 0:
                            raise RuntimeError("invalid # params=%d" % (nParams,))
                        return retVal
                order += 1
        
        numCoeffsList = (1, 3, 6, 10)
        maxCoeffs = numCoeffsList[-1]
        deltaCoeff = 0.3
        allCoeffs = numpy.arange(deltaCoeff, deltaCoeff * (maxCoeffs + 1) + (deltaCoeff / 2.0), deltaCoeff)
        
        for numCoeffs in numCoeffsList:
            coeffs = allCoeffs[0: numCoeffs]
            f = fw.PolynomialFunction2D(coeffs)
            for x in numpy.arange(-10.0, 10.1, 2.5):
                for y in numpy.arange(-10.0, 10.1, 2.5):
                    predVal = basic2DPoly(x, y, coeffs)
                    if not numpy.allclose(predVal, f(x,y)):
                        self.fail("%s = %s != %s for x=%s, y=%s, coeffs=%s" % \
                            (f.__class__.__name__, f(x,y), predVal, x, y, coeffs))
        
        # test that the number of parameters is correct for the given order
        def numParamsFromOrder(order):
            return (order + 1) * (order + 2) / 2
        MaxOrder = 13
        for order in range(MaxOrder+1):
            f = fw.PolynomialFunction2D(order)
            predNParams = numParamsFromOrder(order)
            self.assertEqual(f.getNParameters(), predNParams)
            fw.PolynomialFunction2D(numpy.zeros(predNParams, dtype=float))
        
        # test that the wrong number of parameters raises an exception
        validNumParams = set()
        for order in range(MaxOrder+1):
            validNumParams.add(numParamsFromOrder(order))
        for numParams in range(numParamsFromOrder(MaxOrder)):
            if numParams in validNumParams:
                continue
            self.assertRaises(Exception, fw.PolynomialFunction2D, numpy.zeros(numParams, dtype=float))


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(FunctionTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)

    return unittest.TestSuite(suites)

if __name__ == "__main__":
    tests.run(suite())
