#
# LSST Data Management System
# Copyright 2017 LSST/AURA.
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

import unittest

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.geom
import lsst.afw.image
import lsst.afw.table


def makeTestCurve(random, x0, x1, y0=0.0, y1=0.0):
    """Return a piecewise callable appropriate for testing
    TransmissionCurve objects.

    The returned callable can be called with either scalar or numpy.ndarray
    arguments.

    Between x0 and x1, the returned function is a nonnegative 4th-order
    polynomial.  At and below x0 it is constant with value y0, and at and
    above x1 it is equal to y1.  At exactly x0 and x1 its first derivative
    is zero.

    y0 and y1 may be None to indicate a random value should be drawn
    (on the interval [0, 1]).
    """
    if y0 is None:
        y0 = random.rand()
    if y1 is None:
        y1 = random.rand()
    assert y0 >= 0.0
    assert y1 >= 0.0
    alpha0 = np.abs(y0 - y1)
    if alpha0 == 0.0:
        alpha0 = 1.0
    mu = (x1 - x0)*(0.25 + 0.5*random.rand())
    alpha = alpha0*(0.25 + 0.5*random.rand())
    n = 5
    A = np.zeros([n, n], dtype=float)
    dx = x1 - x0
    A[0, 0] = 1.0
    A[1, :] = [dx**k for k in range(n)]
    A[2, 1] = 1.0
    A[3, :] = [k*dx**(k - 1) for k in range(n)]
    A[4, :] = [mu**k for k in range(n)]
    b = np.array([y0, y1, 0.0, 0.0, alpha], dtype=float)
    coeffs = np.linalg.solve(A, b)

    def curve(x):
        result = sum(c*(x - x0)**k for k, c in enumerate(coeffs))
        result[x <= x0] = y0
        result[x >= x1] = y1
        return result

    return curve


class TransmissionCurveTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        self.random = np.random.RandomState(1)
        self.points = [lsst.afw.geom.Point2D(self.random.rand(), self.random.rand()) for i in range(5)]
        self.minWavelength = 5000 + self.random.rand()
        self.maxWavelength = 5500 + self.random.rand()

    def randIfNone(self, v):
        """Return a random number if the given input is None, but pass it through if it is not."""
        if v is None:
            return self.random.rand()
        return v

    def checkEvaluation(self, tc, wavelengths, expected, rtol=0.0, atol=0.0):
        """Test that evaluating a TransmissionCurve on the given wavelengths array yields the given
        expected throughputs.
        """
        for point in self.points:
            throughput = tc.sampleAt(point, wavelengths)
            self.assertFloatsAlmostEqual(throughput, expected, rtol=rtol, atol=atol)
            throughput2 = np.zeros(wavelengths.size, dtype=float)
            tc.sampleAt(point, wavelengths, out=throughput2)
            self.assertFloatsEqual(throughput2, throughput)

    def assertTransmissionCurvesEqual(self, a, b, rtol=0.0, atol=0.0):
        """Test whether two TransimssionCurves are equivalent."""
        self.assertEqual(a.getWavelengthBounds(), b.getWavelengthBounds())
        self.assertEqual(a.getThroughputAtBounds(), b.getThroughputAtBounds())
        wavelengths = np.linspace(*(a.getWavelengthBounds() + (100,)))
        for point in self.points:
            self.assertFloatsAlmostEqual(
                a.sampleAt(point, wavelengths),
                b.sampleAt(point, wavelengths),
                rtol=rtol, atol=atol
            )

    def checkPersistence(self, tc, points=None):
        """Test that a TransmissionCurve round-trips through persistence."""
        if points is None:
            points = self.points
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            tc.writeFits(filename)
            tc2 = lsst.afw.image.TransmissionCurve.readFits(filename)
        self.assertTransmissionCurvesEqual(tc, tc2)

    def makeAndCheckSpatiallyConstant(self, curve, wavelengths, throughputAtMin, throughputAtMax):
        """Construct a constant TransmissionCurve and apply basic tests to it."""
        throughput = curve(wavelengths)
        tc = lsst.afw.image.TransmissionCurve.makeSpatiallyConstant(throughput, wavelengths,
                                                                    throughputAtMin, throughputAtMax)
        self.assertEqual(tc.getWavelengthBounds(), (wavelengths[0], wavelengths[-1]))
        self.assertEqual(tc.getThroughputAtBounds(), (throughputAtMin, throughputAtMax))
        self.checkEvaluation(tc, wavelengths, throughput)
        self.checkEvaluation(tc, np.array([2*wavelengths[0] - wavelengths[1]]), throughputAtMin)
        self.checkEvaluation(tc, np.array([2*wavelengths[-1] - wavelengths[-2]]), throughputAtMax)
        # Check that we get decent results when interpolating to different wavelengths.
        wavelengths1 = np.linspace(wavelengths[0] - 10, wavelengths[-1] + 10, 200)
        self.checkEvaluation(tc, wavelengths1, curve(wavelengths1), rtol=1E-2, atol=1E-2)
        # Test that multiplication with identity is a no-op
        tc2 = tc * lsst.afw.image.TransmissionCurve.makeIdentity()
        self.assertTransmissionCurvesEqual(tc, tc2)
        return tc

    def checkSpatiallyConstantEvenSpacing(self, throughputAtMin, throughputAtMax):
        """Test that we can construct and use a spatially-constant
        TransmissionCurve initialized with an evenly-spaced wavelength
        array.
        """
        throughputAtMin = self.randIfNone(throughputAtMin)
        throughputAtMax = self.randIfNone(throughputAtMax)
        wavelengths = np.linspace(self.minWavelength, self.maxWavelength, 100)
        curve = makeTestCurve(self.random, self.minWavelength, self.maxWavelength,
                              throughputAtMin, throughputAtMax)
        tc = self.makeAndCheckSpatiallyConstant(curve, wavelengths, throughputAtMin, throughputAtMax)
        self.checkPersistence(tc)

    def testSpatiallyConstantEvenSpacing(self):
        """Invoke all SpatiallyConstantEvenSpacing tests.

        Should be updated to use subTest when Python < 3.4 support is ended.
        """
        self.checkSpatiallyConstantEvenSpacing(0.0, 0.0)
        self.checkSpatiallyConstantEvenSpacing(0.0, None)
        self.checkSpatiallyConstantEvenSpacing(None, 0.0)
        self.checkSpatiallyConstantEvenSpacing(None, None)

    def checkSpatiallyConstantUnevenSpacing(self, throughputAtMin, throughputAtMax):
        """Test that we can construct and use a spatially-constant
        TransmissionCurve initialized with an unevenly-spaced wavelength
        array.
        """
        throughputAtMin = self.randIfNone(throughputAtMin)
        throughputAtMax = self.randIfNone(throughputAtMax)
        wavelengths = self.minWavelength + (self.maxWavelength - self.minWavelength)*self.random.rand(100)
        wavelengths.sort()
        curve = makeTestCurve(self.random, self.minWavelength, self.maxWavelength,
                              throughputAtMin, throughputAtMax)
        tc = self.makeAndCheckSpatiallyConstant(curve, wavelengths, throughputAtMin, throughputAtMax)
        self.checkPersistence(tc)

    def testSpatiallyConstantUnevenSpacing(self):
        """Invoke all SpatiallyConstantUnevenSpacing tests.

        Should be updated to use subTest when Python < 3.4 support is ended.
        """
        self.checkSpatiallyConstantUnevenSpacing(0.0, 0.0)
        self.checkSpatiallyConstantUnevenSpacing(0.0, None)
        self.checkSpatiallyConstantUnevenSpacing(None, 0.0)
        self.checkSpatiallyConstantUnevenSpacing(None, None)

    def checkProduct(self, throughputAtMin1, throughputAtMax1, throughputAtMin2, throughputAtMax2):
        """Test the product of two spatially-constant TransmissionCurves."""
        throughputAtMin1 = self.randIfNone(throughputAtMin1)
        throughputAtMax1 = self.randIfNone(throughputAtMax1)
        throughputAtMin2 = self.randIfNone(throughputAtMin2)
        throughputAtMax2 = self.randIfNone(throughputAtMax2)
        wl1a = 5100 + self.random.rand()
        wl1b = 5500 + self.random.rand()
        wl2a = 5200 + self.random.rand()
        wl2b = 5600 + self.random.rand()
        wl1 = np.linspace(wl1a, wl1b, 100)
        wl2 = np.linspace(wl2a, wl2b, 100)
        curve1 = makeTestCurve(self.random, wl1a, wl1b, throughputAtMin1, throughputAtMax1)
        curve2 = makeTestCurve(self.random, wl2a, wl2b, throughputAtMin2, throughputAtMax2)
        op1 = self.makeAndCheckSpatiallyConstant(curve1, wl1, throughputAtMin1, throughputAtMax1)
        op2 = self.makeAndCheckSpatiallyConstant(curve2, wl2, throughputAtMin2, throughputAtMax2)
        product = op1 * op2

        lowest = np.linspace(wl1a - 10, wl1a - 1, 10)
        self.checkEvaluation(product, lowest, throughputAtMin1*throughputAtMin2)

        lower = np.linspace(wl1a + 1, wl2a - 1, 10)
        if throughputAtMin2 == 0.0:
            self.checkEvaluation(product, lower, 0.0)
        else:
            for point in self.points:
                self.assertFloatsEqual(
                    op1.sampleAt(point, lower)*throughputAtMin2,
                    product.sampleAt(point, lower)
                )

        inner = np.linspace(wl2a, wl1b, 10)
        for point in self.points:
            self.assertFloatsAlmostEqual(
                op1.sampleAt(point, inner)*op2.sampleAt(point, inner),
                product.sampleAt(point, inner)
            )

        upper = np.linspace(wl1b + 1, wl2b - 1, 10)
        if throughputAtMax1 == 0.0:
            self.checkEvaluation(product, upper, 0.0)
        else:
            for point in self.points:
                self.assertFloatsEqual(
                    op2.sampleAt(point, upper)*throughputAtMax1,
                    product.sampleAt(point, upper)
                )

        uppermost = np.linspace(wl2b + 1, wl2b + 10, 10)
        self.checkEvaluation(product, uppermost, throughputAtMax1*throughputAtMax2)

        self.checkPersistence(product)

    def testProduct(self):
        """Invoke all checkProduct tests.

        Should be updated to use subTest when Python < 3.4 support is ended.
        """
        self.checkProduct(0.0, 0.0, 0.0, 0.0)
        self.checkProduct(0.0, 0.0, 0.0, None)
        self.checkProduct(0.0, 0.0, None, 0.0)
        self.checkProduct(0.0, 0.0, None, None)
        self.checkProduct(0.0, None, 0.0, 0.0)
        self.checkProduct(0.0, None, 0.0, None)
        self.checkProduct(0.0, None, None, 0.0)
        self.checkProduct(0.0, None, None, None)
        self.checkProduct(None, 0.0, 0.0, 0.0)
        self.checkProduct(None, 0.0, 0.0, None)
        self.checkProduct(None, 0.0, None, 0.0)
        self.checkProduct(None, 0.0, None, None)
        self.checkProduct(None, None, 0.0, 0.0)
        self.checkProduct(None, None, 0.0, None)
        self.checkProduct(None, None, None, 0.0)
        self.checkProduct(None, None, None, None)

    def makeRadial(self):
        """Construct a random radial TransmissionCurve and return it with
        the wavelengths, radii, and 2-d curve used to construct it.
        """
        wavelengths = np.linspace(self.minWavelength, self.maxWavelength, 100)
        radii = np.linspace(0.0, 1.0, 200)

        # This curve will represent the TransmissionCurve at the origin;
        # we'll shift it to higher wavelengths and scale it linearly with radius
        curve = makeTestCurve(self.random, wavelengths[0], wavelengths[90])
        delta = (wavelengths[1] - wavelengths[0])/(radii[1] - radii[0])

        def curve2d(lam, r):
            return curve(lam + delta*r)*(1.0+r)

        throughput = np.zeros(wavelengths.shape + radii.shape, dtype=float)
        for i, radius in enumerate(radii):
            throughput[:, i] = curve2d(wavelengths, radius)

        tc = lsst.afw.image.TransmissionCurve.makeRadial(throughput, wavelengths, radii)

        return tc, wavelengths, radii, curve2d

    def testRadial(self):
        """Test the functionality of radial TransmissionCurves."""
        tc, wavelengths, radii, curve2d = self.makeRadial()

        # Test at exactly the radii and wavelengths we initialized with.
        for n, radius in enumerate(radii):
            rot = lsst.afw.geom.LinearTransform.makeRotation(
                2.0*np.pi*self.random.rand()*lsst.afw.geom.radians
            )
            p0 = lsst.afw.geom.Point2D(0.0, radius)
            p1 = rot(p0)
            self.assertFloatsAlmostEqual(
                curve2d(wavelengths, radius),
                tc.sampleAt(p0, wavelengths),
                rtol=1E-13
            )
            self.assertFloatsAlmostEqual(
                curve2d(wavelengths, radius),
                tc.sampleAt(p1, wavelengths),
                rtol=1E-13
            )

        # Test at some other random points in radius and wavelength.
        wl2 = np.linspace(self.minWavelength, self.maxWavelength, 151)
        for point in self.points:
            radius = (point.getX()**2 + point.getY()**2)**0.5
            self.assertFloatsAlmostEqual(
                curve2d(wl2, radius),
                tc.sampleAt(point, wl2),
                rtol=1E-2, atol=1E-2
            )

        # Test persistence for radial TransmissionCurves
        self.checkPersistence(tc)

    def testTransform(self):
        """Test that we can transform a spatially-varying TransmissionCurve."""
        tc, wavelengths, radii, curve2d = self.makeRadial()

        # If we transform by a pure rotation, what we get back should be equivalent.
        affine1 = lsst.afw.geom.AffineTransform(
            lsst.afw.geom.LinearTransform.makeRotation(
                2.0*np.pi*self.random.rand()*lsst.afw.geom.radians
            )
        )
        transform1 = lsst.afw.geom.makeTransform(affine1)
        wl2 = np.linspace(self.minWavelength, self.maxWavelength, 151)
        tc1 = tc.transformedBy(transform1)
        self.assertTransmissionCurvesEqual(tc, tc1, rtol=1E-13)

        # Test transforming by a random affine transform.
        affine2 = lsst.afw.geom.AffineTransform(
            lsst.afw.geom.LinearTransform.makeScaling(1.0 + self.random.rand()),
            lsst.afw.geom.Extent2D(self.random.randn(), self.random.randn())
        )
        transform2 = lsst.afw.geom.makeTransform(affine2)
        tc2 = tc.transformedBy(transform2)
        for point in self.points:
            self.assertFloatsAlmostEqual(
                tc.sampleAt(point, wl2),
                tc2.sampleAt(affine2(point), wl2),
                rtol=1E-13
            )

        # Test further transforming the rotated transmission curve
        tc3 = tc1.transformedBy(transform2)
        self.assertTransmissionCurvesEqual(tc2, tc3)

        # Test persistence for transformed TransmissionCurves
        self.checkPersistence(tc3)

    def testExposure(self):
        """Test that we can attach a TransmissionCurve to an Exposure and round-trip it through I/O."""
        wavelengths = np.linspace(6200, 6400, 100)
        curve = makeTestCurve(self.random, 6200, 6400, 0.0, 0.0)
        tc1 = self.makeAndCheckSpatiallyConstant(curve, wavelengths, 0.0, 0.0)
        exposure1 = lsst.afw.image.ExposureF(4, 5)
        exposure1.getInfo().setTransmissionCurve(tc1)
        self.assertTrue(exposure1.getInfo().hasTransmissionCurve())
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            exposure1.writeFits(filename)
            exposure2 = lsst.afw.image.ExposureF(filename)
        self.assertTrue(exposure2.getInfo().hasTransmissionCurve())
        tc2 = exposure2.getInfo().getTransmissionCurve()
        self.assertTransmissionCurvesEqual(tc1, tc2)

    def testExposureRecord(self):
        """Test that we can attach a TransmissionCurve to an ExposureRecord and round-trip it through I/O."""
        wavelengths = np.linspace(6200, 6400, 100)
        curve = makeTestCurve(self.random, 6200, 6400, 0.0, 0.0)
        tc1 = self.makeAndCheckSpatiallyConstant(curve, wavelengths, 0.0, 0.0)
        cat1 = lsst.afw.table.ExposureCatalog(lsst.afw.table.ExposureTable.makeMinimalSchema())
        cat1.addNew().setTransmissionCurve(tc1)
        self.assertTrue(cat1[0].getTransmissionCurve() is not None)
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            cat1.writeFits(filename)
            cat2 = lsst.afw.table.ExposureCatalog.readFits(filename)
        self.assertTrue(cat2[0].getTransmissionCurve() is not None)
        tc2 = cat2[0].getTransmissionCurve()
        self.assertTransmissionCurvesEqual(tc1, tc2)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
