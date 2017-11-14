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
from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.geom
import lsst.afw.image

try:
    type(display)
except NameError:
    display = False


def makeTestCurve(random, x0, x1, y0=0.0, y1=0.0):
    """Return a piecewise NumPy-friendly callable appropriate for testing
    TransmissionCurve objects.

    Between x0 and x1, the returned function is a nonnegative 4th-order
    polynomial.  At and below x0 it is constant with value y0, and at and
    above x1 it is equal to y1.  At exactly x0 and x1 its first derivative
    is zero.

    y0 and y1 may be None or NaN to indicate a random value should be drawn
    (on the interval [0, 1]).
    """
    if y0 is None or np.isnan(y0):
        y0 = random.rand()
    if y1 is None or np.isnan(y1):
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


def useOutArg(callable, size, *args):
    """Helper for testing callables that fill arrays via an 'out' argument."""
    out = np.zeros(size, dtype=float)
    callable(*args, out=out)
    return out


class TransmissionCurveTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        self.random = np.random.RandomState(1)
        self.point = lsst.afw.geom.Point2D(0.0, 0.0)

    def assertEqualOrNaN(self, a, b):
        if not (np.isnan(a) and np.isnan(b)):
            self.assertEqual(a, b)

    def checkPersistence(self, tc, points=None):
        """Test that a TransmissionCurve round-trips through persistence."""
        if points is None:
            points = [self.point]
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            tc.writeFits(filename)
            tc2 = lsst.afw.image.TransmissionCurve.readFits(filename)
        self.assertEqual(
            tc.getNaturalSampling().min,
            tc2.getNaturalSampling().min
        )
        self.assertEqual(
            tc.getNaturalSampling().max,
            tc2.getNaturalSampling().max
        )
        self.assertEqual(
            tc.getNaturalSampling().size,
            tc2.getNaturalSampling().size
        )
        self.assertEqualOrNaN(tc.getThroughputAtBounds()[0], tc2.getThroughputAtBounds()[0])
        self.assertEqualOrNaN(tc.getThroughputAtBounds()[1], tc2.getThroughputAtBounds()[1])
        for point in points:
            self.assertFloatsEqual(
                tc.sampleAt(point, tc.getNaturalSampling()),
                tc2.sampleAt(point, tc.getNaturalSampling())
            )

    def makeAndCheckConstant(self, curve, wavelengths, throughputAtMin, throughputAtMax):
        """Construct a constant TransmissionCurve and apply basic tests to it."""
        throughput = curve(wavelengths)
        tc = lsst.afw.image.TransmissionCurve.makeConstant(throughput, wavelengths,
                                                           throughputAtMin, throughputAtMax)
        self.assertEqual(tc.getNaturalSampling().min, wavelengths[0])
        self.assertEqual(tc.getNaturalSampling().max, wavelengths[-1])
        self.assertFloatsEqual(
            throughput,
            tc.sampleAt(self.point, wavelengths)
        )
        self.assertFloatsEqual(
            throughput,
            useOutArg(tc.sampleAt, wavelengths.size, self.point, wavelengths)
        )
        if not np.isnan(throughputAtMin):
            self.assertEqual(tc.getThroughputAtBounds()[0], throughputAtMin)
            self.assertFloatsEqual(
                throughputAtMin,
                tc.sampleAt(self.point, np.array([2*wavelengths[0] - wavelengths[1]]))
            )
        else:
            self.assertTrue(np.isnan(tc.getThroughputAtBounds()[0]))
        if not np.isnan(throughputAtMax):
            self.assertEqual(tc.getThroughputAtBounds()[1], throughputAtMax)
            self.assertFloatsEqual(
                throughputAtMax,
                tc.sampleAt(self.point, np.array([2*wavelengths[-1] - wavelengths[-2]]))
            )
        else:
            self.assertTrue(np.isnan(tc.getThroughputAtBounds()[1]))
        # Check that we get decent results when interpolating to different wavelengths.
        wavelengths1 = np.linspace(wavelengths[0] - 10, wavelengths[-1] + 10, 200)
        if np.isnan(throughputAtMin):
            if np.isnan(throughputAtMax):
                mask = np.logical_and(wavelengths1 >= wavelengths[0], wavelengths1 <= wavelengths[-1])
            else:
                mask = wavelengths1 >= wavelengths[0]
        else:
            if np.isnan(throughputAtMax):
                mask = wavelengths1 <= wavelengths[-1]
            else:
                mask = np.ones(wavelengths1.size, dtype=bool)
        self.assertFloatsAlmostEqual(
            curve(wavelengths1)[mask],
            tc.sampleAt(self.point, wavelengths1)[mask],
            rtol=1E-2, atol=1E-2
        )
        if mask.any():
            self.assertTrue(
                np.isnan(tc.sampleAt(self.point, wavelengths1)[np.logical_not(mask)]).all()
            )
        # Test that multiplication with identity is a no-op
        tc2 = tc * lsst.afw.image.TransmissionCurve.makeIdentity()
        self.assertFloatsEqual(
            tc.sampleAt(self.point, wavelengths1)[mask],
            tc2.sampleAt(self.point, wavelengths1)[mask]
        )
        return tc

    def checkConstantEvenSpacing(self, minWavelength, maxWavelength, throughputAtMin, throughputAtMax):
        """Test that we can construct and use a spatially-constant
        TransmissionCurve initialized with an evenly-spaced wavelength
        array.
        """
        wavelengths = np.linspace(minWavelength, maxWavelength, 100)
        curve = makeTestCurve(self.random, minWavelength, maxWavelength, throughputAtMin, throughputAtMax)
        tc = self.makeAndCheckConstant(curve, wavelengths, throughputAtMin, throughputAtMax)
        self.assertEqual(tc.getNaturalSampling().size, wavelengths.size)
        self.assertFloatsAlmostEqual(
            wavelengths,
            tc.getNaturalSampling().makeArray(),
            rtol=1E-14
        )
        self.assertFloatsAlmostEqual(
            curve(wavelengths),
            tc.sampleAt(self.point, tc.getNaturalSampling()),
            rtol=1E-11, atol=1E-11
        )
        self.assertFloatsAlmostEqual(
            curve(wavelengths),
            useOutArg(tc.sampleAt, wavelengths.size, self.point, tc.getNaturalSampling()),
            rtol=1E-11, atol=1E-11
        )
        self.checkPersistence(tc)

    def testConstantEvenSpacing(self):
        # Don't want to accidentally test floating-point ops with numbers
        # that can be exactly represented, so we add random values
        minWavelength = 6000 + self.random.randn()
        maxWavelength = 6200 + self.random.randn()
        nan = float("nan")
        self.checkConstantEvenSpacing(minWavelength, maxWavelength, 0.0, 0.0)
        self.checkConstantEvenSpacing(minWavelength, maxWavelength, 0.0, 1.0)
        self.checkConstantEvenSpacing(minWavelength, maxWavelength, 0.0, nan)
        self.checkConstantEvenSpacing(minWavelength, maxWavelength, 1.0, 0.0)
        self.checkConstantEvenSpacing(minWavelength, maxWavelength, 1.0, 1.0)
        self.checkConstantEvenSpacing(minWavelength, maxWavelength, 1.0, nan)
        self.checkConstantEvenSpacing(minWavelength, maxWavelength, nan, 0.0)
        self.checkConstantEvenSpacing(minWavelength, maxWavelength, nan, 1.0)
        self.checkConstantEvenSpacing(minWavelength, maxWavelength, nan, nan)

    def checkConstantUnevenSpacing(self, minWavelength, maxWavelength, throughputAtMin, throughputAtMax):
        """Test that we can construct and use a spatially-constant
        TransmissionCurve initialized with an unevenly-spaced wavelength
        array.
        """
        wavelengths = minWavelength + (maxWavelength - minWavelength)*self.random.rand(100)
        wavelengths.sort()
        curve = makeTestCurve(self.random, minWavelength, maxWavelength, throughputAtMin, throughputAtMax)
        tc = self.makeAndCheckConstant(curve, wavelengths, throughputAtMin, throughputAtMax)
        self.assertLess(tc.getNaturalSampling().getSpacing(), min(wavelengths[1:] - wavelengths[:-1]))
        self.checkPersistence(tc)

    def testConstantUnevenSpacing(self):
        # Don't want to accidentally test floating-point ops with numbers
        # that can be exactly represented, so we add random values
        minWavelength = 7200 + self.random.randn()
        maxWavelength = 7400 + self.random.randn()
        nan = float("nan")
        self.checkConstantUnevenSpacing(minWavelength, maxWavelength, 0.0, 0.0)
        self.checkConstantUnevenSpacing(minWavelength, maxWavelength, 0.0, 1.0)
        self.checkConstantUnevenSpacing(minWavelength, maxWavelength, 0.0, nan)
        self.checkConstantUnevenSpacing(minWavelength, maxWavelength, 1.0, 0.0)
        self.checkConstantUnevenSpacing(minWavelength, maxWavelength, 1.0, 1.0)
        self.checkConstantUnevenSpacing(minWavelength, maxWavelength, 1.0, nan)
        self.checkConstantUnevenSpacing(minWavelength, maxWavelength, nan, 0.0)
        self.checkConstantUnevenSpacing(minWavelength, maxWavelength, nan, 1.0)
        self.checkConstantUnevenSpacing(minWavelength, maxWavelength, nan, nan)

    def checkProduct(self, throughputAtMin1, throughputAtMax1, throughputAtMin2, throughputAtMax2):
        wl1a = 5100 + self.random.rand()
        wl1b = 5500
        wl2a = 5200
        wl2b = 5600 + self.random.rand()
        wl1 = np.linspace(wl1a, wl1b, 100)
        wl2 = np.linspace(wl2a, wl2b, 100)
        curve1 = makeTestCurve(self.random, wl1a, wl1b, throughputAtMin1, throughputAtMax1)
        curve2 = makeTestCurve(self.random, wl2a, wl2b, throughputAtMin2, throughputAtMax2)
        op1 = self.makeAndCheckConstant(curve1, wl1, throughputAtMin1, throughputAtMax1)
        op2 = self.makeAndCheckConstant(curve2, wl2, throughputAtMin2, throughputAtMax2)
        product = op1 * op2
        self.assertLessEqual(
            product.getNaturalSampling().getSpacing(),
            op1.getNaturalSampling().getSpacing()
        )
        self.assertLessEqual(
            product.getNaturalSampling().getSpacing(),
            op2.getNaturalSampling().getSpacing()
        )

        lowest = product.SampleDef(wl1a - 10, wl1a - 1, 10)
        if throughputAtMin1 == 0.0 or throughputAtMin2 == 0.0:
            self.assertFloatsEqual(
                product.sampleAt(self.point, lowest),
                0.0
            )
        elif np.isnan(throughputAtMin1) or np.isnan(throughputAtMin2):
            self.assertTrue(np.isnan(product.sampleAt(self.point, lowest)).all())
        else:
            self.assertFloatsEqual(
                op1.sampleAt(self.point, lowest),
                throughputAtMin1*throughputAtMin2
            )

        lower = product.SampleDef(wl1a + 1, wl2a - 1, 10)
        if throughputAtMin2 == 0.0:
            self.assertFloatsEqual(
                product.sampleAt(self.point, lower),
                0.0
            )
        elif np.isnan(throughputAtMin2):
            self.assertTrue(np.isnan(product.sampleAt(self.point, lower)).all())
        else:
            self.assertFloatsEqual(
                op1.sampleAt(self.point, lower)*throughputAtMin2,
                product.sampleAt(self.point, lower)
            )

        inner = product.SampleDef(wl2a, wl1b, 10)
        self.assertFloatsAlmostEqual(
            op1.sampleAt(self.point, inner)*op2.sampleAt(self.point, inner),
            product.sampleAt(self.point, inner)
        )

        upper = product.SampleDef(wl1b + 1, wl2b - 1, 10)
        if throughputAtMax1 == 0.0:
            self.assertFloatsEqual(
                product.sampleAt(self.point, upper),
                0.0
            )
        elif np.isnan(throughputAtMax1):
            self.assertTrue(np.isnan(product.sampleAt(self.point, upper)).all())
        else:
            self.assertFloatsEqual(
                op2.sampleAt(self.point, upper)*throughputAtMax1,
                product.sampleAt(self.point, upper)
            )

        uppermost = product.SampleDef(wl2b + 1, wl2b + 10, 10)
        if throughputAtMax1 == 0.0 or throughputAtMax2 == 0.0:
            self.assertFloatsEqual(
                product.sampleAt(self.point, uppermost),
                0.0
            )
        elif np.isnan(throughputAtMax1) or np.isnan(throughputAtMax2):
            self.assertTrue(np.isnan(product.sampleAt(self.point, uppermost)).all())
        else:
            self.assertFloatsEqual(
                op1.sampleAt(self.point, uppermost),
                throughputAtMax1*throughputAtMax2
            )

        self.checkPersistence(product)

    def testProduct(self):
        nan = float('nan')
        # In Python 3.4+, we could replace this with a nested loop and subTest
        # context managers, but for now this seems like the best way to avoid
        # the problem with knowing which iteration failed in a loop.
        self.checkProduct(0.0, 0.0, 0.0, 0.0)
        self.checkProduct(0.0, 0.0, 0.0, 1.0)
        self.checkProduct(0.0, 0.0, 0.0, nan)
        self.checkProduct(0.0, 0.0, 1.0, 0.0)
        self.checkProduct(0.0, 0.0, 1.0, 1.0)
        self.checkProduct(0.0, 0.0, 1.0, nan)
        self.checkProduct(0.0, 0.0, nan, 0.0)
        self.checkProduct(0.0, 0.0, nan, 1.0)
        self.checkProduct(0.0, 0.0, nan, nan)
        self.checkProduct(0.0, 1.0, 0.0, 0.0)
        self.checkProduct(0.0, 1.0, 0.0, 1.0)
        self.checkProduct(0.0, 1.0, 0.0, nan)
        self.checkProduct(0.0, 1.0, 1.0, 0.0)
        self.checkProduct(0.0, 1.0, 1.0, 1.0)
        self.checkProduct(0.0, 1.0, 1.0, nan)
        self.checkProduct(0.0, 1.0, nan, 0.0)
        self.checkProduct(0.0, 1.0, nan, 1.0)
        self.checkProduct(0.0, 1.0, nan, nan)
        self.checkProduct(0.0, nan, 0.0, 0.0)
        self.checkProduct(0.0, nan, 0.0, 1.0)
        self.checkProduct(0.0, nan, 0.0, nan)
        self.checkProduct(0.0, nan, 1.0, 0.0)
        self.checkProduct(0.0, nan, 1.0, 1.0)
        self.checkProduct(0.0, nan, 1.0, nan)
        self.checkProduct(0.0, nan, nan, 0.0)
        self.checkProduct(0.0, nan, nan, 1.0)
        self.checkProduct(0.0, nan, nan, nan)
        self.checkProduct(1.0, 0.0, 0.0, 0.0)
        self.checkProduct(1.0, 0.0, 0.0, 1.0)
        self.checkProduct(1.0, 0.0, 0.0, nan)
        self.checkProduct(1.0, 0.0, 1.0, 0.0)
        self.checkProduct(1.0, 0.0, 1.0, 1.0)
        self.checkProduct(1.0, 0.0, 1.0, nan)
        self.checkProduct(1.0, 0.0, nan, 0.0)
        self.checkProduct(1.0, 0.0, nan, 1.0)
        self.checkProduct(1.0, 0.0, nan, nan)
        self.checkProduct(1.0, 1.0, 0.0, 0.0)
        self.checkProduct(1.0, 1.0, 0.0, 1.0)
        self.checkProduct(1.0, 1.0, 0.0, nan)
        self.checkProduct(1.0, 1.0, 1.0, 0.0)
        self.checkProduct(1.0, 1.0, 1.0, 1.0)
        self.checkProduct(1.0, 1.0, 1.0, nan)
        self.checkProduct(1.0, 1.0, nan, 0.0)
        self.checkProduct(1.0, 1.0, nan, 1.0)
        self.checkProduct(1.0, 1.0, nan, nan)
        self.checkProduct(1.0, nan, 0.0, 0.0)
        self.checkProduct(1.0, nan, 0.0, 1.0)
        self.checkProduct(1.0, nan, 0.0, nan)
        self.checkProduct(1.0, nan, 1.0, 0.0)
        self.checkProduct(1.0, nan, 1.0, 1.0)
        self.checkProduct(1.0, nan, 1.0, nan)
        self.checkProduct(1.0, nan, nan, 0.0)
        self.checkProduct(1.0, nan, nan, 1.0)
        self.checkProduct(1.0, nan, nan, nan)
        self.checkProduct(nan, 0.0, 0.0, 0.0)
        self.checkProduct(nan, 0.0, 0.0, 1.0)
        self.checkProduct(nan, 0.0, 0.0, nan)
        self.checkProduct(nan, 0.0, 1.0, 0.0)
        self.checkProduct(nan, 0.0, 1.0, 1.0)
        self.checkProduct(nan, 0.0, 1.0, nan)
        self.checkProduct(nan, 0.0, nan, 0.0)
        self.checkProduct(nan, 0.0, nan, 1.0)
        self.checkProduct(nan, 0.0, nan, nan)
        self.checkProduct(nan, 1.0, 0.0, 0.0)
        self.checkProduct(nan, 1.0, 0.0, 1.0)
        self.checkProduct(nan, 1.0, 0.0, nan)
        self.checkProduct(nan, 1.0, 1.0, 0.0)
        self.checkProduct(nan, 1.0, 1.0, 1.0)
        self.checkProduct(nan, 1.0, 1.0, nan)
        self.checkProduct(nan, 1.0, nan, 0.0)
        self.checkProduct(nan, 1.0, nan, 1.0)
        self.checkProduct(nan, 1.0, nan, nan)
        self.checkProduct(nan, nan, 0.0, 0.0)
        self.checkProduct(nan, nan, 0.0, 1.0)
        self.checkProduct(nan, nan, 0.0, nan)
        self.checkProduct(nan, nan, 1.0, 0.0)
        self.checkProduct(nan, nan, 1.0, 1.0)
        self.checkProduct(nan, nan, 1.0, nan)
        self.checkProduct(nan, nan, nan, 0.0)
        self.checkProduct(nan, nan, nan, 1.0)
        self.checkProduct(nan, nan, nan, nan)

    def testRadialAndTransform(self):
        minWavelength = 6000 + self.random.randn()
        maxWavelength = 6200 + self.random.randn()
        wavelengths = np.linspace(minWavelength, maxWavelength, 100)
        radii = np.linspace(0.0, 1.0, 10)

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

        # Test at exactly the radii and wavelengths we initialized with.
        for n, radius in enumerate(radii):
            rot = lsst.afw.geom.LinearTransform.makeRotation(
                2.0*np.pi*self.random.rand()*lsst.afw.geom.radians
            )
            p0 = lsst.afw.geom.Point2D(0.0, radius)
            p1 = rot(p0)
            self.assertFloatsAlmostEqual(
                throughput[:, n],
                tc.sampleAt(p0, wavelengths),
                rtol=1E-15
            )
            self.assertFloatsAlmostEqual(
                throughput[:, n],
                tc.sampleAt(p1, wavelengths),
                rtol=1E-15
            )

        # Test at some other random points in radius and wavelength.
        points = []
        for n in range(10):
            rot = lsst.afw.geom.LinearTransform.makeRotation(
                2.0*np.pi*self.random.rand()*lsst.afw.geom.radians
            )
            radius = self.random.rand()
            p0 = lsst.afw.geom.Point2D(0.0, radius)
            p1 = rot(p0)
            points.append(p1)
            wl2 = np.linspace(minWavelength, maxWavelength, 151)
            self.assertFloatsAlmostEqual(
                curve2d(wl2, radius),
                tc.sampleAt(p0, wl2),
                rtol=1E-2, atol=1E-2
            )
            self.assertFloatsAlmostEqual(
                curve2d(wl2, radius),
                tc.sampleAt(p1, wl2),
                rtol=1E-2, atol=1E-2
            )
            try:
                pass
            except AssertionError:
                from matplotlib import pyplot
                pyplot.plot(wl2, curve2d(wl2, radius), 'r')
                pyplot.plot(wl2, tc.sampleAt(p0, wl2), 'b:')
                pyplot.plot(wl2, tc.sampleAt(p1, wl2), 'c:')
                pyplot.axvline(minWavelength, color='k')
                pyplot.axvline(maxWavelength, color='k')
                pyplot.show()
                raise

        # Test persistence for radial TransmissionCurves
        self.checkPersistence(tc, points)

        # If we transform by a pure rotation, what we get back should be equivalent.
        affine1 = lsst.afw.geom.AffineTransform(
            lsst.afw.geom.LinearTransform.makeRotation(
                2.0*np.pi*self.random.rand()*lsst.afw.geom.radians
            )
        )
        transform1 = lsst.afw.geom.makeTransform(affine1)
        tc1 = tc.transform(transform1)
        for point in points:
            self.assertFloatsAlmostEqual(
                tc.sampleAt(point, wl2),
                tc1.sampleAt(point, wl2),
                rtol=1E-15
            )

        # Test transforming by a random affine transform.
        affine2 = lsst.afw.geom.AffineTransform(
            lsst.afw.geom.LinearTransform.makeScaling(1.0 + self.random.rand()),
            lsst.afw.geom.Extent2D(self.random.randn(), self.random.randn())
        )
        transform2 = lsst.afw.geom.makeTransform(affine2)
        tc2 = tc.transform(transform2)
        for point in points:
            self.assertFloatsAlmostEqual(
                tc.sampleAt(affine2(point), wl2),
                tc2.sampleAt(point, wl2),
                rtol=1E-15
            )

        # Test further transforming the rotated transmission curve
        tc3 = tc1.transform(transform2)
        for point in points:
            self.assertFloatsAlmostEqual(
                tc2.sampleAt(point, wl2),
                tc3.sampleAt(point, wl2),
                rtol=1E-15
            )

        # Test persistence for transformed TransmissionCurves
        self.checkPersistence(tc3, points)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
