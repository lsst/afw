# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Tests for math.ChebyshevBoundedField

Run with:
   python test_chebyshevBoundedField.py
or
   pytest test_chebyshevBoundedField.py
"""

import os
import unittest

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.geom
import lsst.afw.image
import lsst.afw.math
import lsst.afw.geom

testPath = os.path.abspath(os.path.dirname(__file__))

CHEBYSHEV_T = [
    lambda x: x**0,
    lambda x: x,
    lambda x: 2*x**2 - 1,
    lambda x: (4*x**2 - 3)*x,
    lambda x: (8*x**2 - 8)*x**2 + 1,
    lambda x: ((16*x**2 - 20)*x**2 + 5)*x,
]


def multiply(image, field):
    """Return the product of image and field() at each point in image.
    """
    box = image.getBBox()
    outImage = lsst.afw.image.ImageF(box)
    for i in range(box.getMinX(), box.getMaxX() + 1):
        for j in range(box.getMinY(), box.getMaxY() + 1):
            outImage[i, j] = image[i, j]*field.evaluate(i, j)
    return outImage


def divide(image, field):
    """Return the quotient of image and field() at each point in image.
    """
    box = image.getBBox()
    outImage = lsst.afw.image.ImageF(box)
    for i in range(box.getMinX(), box.getMaxX() + 1):
        for j in range(box.getMinY(), box.getMaxY() + 1):
            outImage[i, j] = image[i, j]/field.evaluate(i, j)
    return outImage


class ChebyshevBoundedFieldTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        self.longMessage = True
        np.random.seed(5)
        self.bbox = lsst.geom.Box2I(
            lsst.geom.Point2I(-5, -5), lsst.geom.Point2I(5, 5))
        self.x1d = np.linspace(self.bbox.getBeginX(), self.bbox.getEndX())
        self.y1d = np.linspace(self.bbox.getBeginY(), self.bbox.getEndY())
        self.x2d, self.y2d = np.meshgrid(self.x1d, self.y1d)
        self.xFlat = np.ravel(self.x2d)
        self.yFlat = np.ravel(self.y2d)
        self.cases = []
        for orderX in range(0, 5):
            for orderY in range(0, 5):
                indexX, indexY = np.meshgrid(np.arange(orderX + 1, dtype=int),
                                             np.arange(orderY + 1, dtype=int))
                for triangular in (True, False):
                    ctrl = lsst.afw.math.ChebyshevBoundedFieldControl()
                    ctrl.orderX = orderX
                    ctrl.orderY = orderY
                    ctrl.triangular = triangular
                    coefficients = np.random.randn(orderY + 1, orderX + 1)
                    if triangular:
                        coefficients[indexX + indexY > max(orderX, orderY)] = 0.0
                    self.cases.append((ctrl, coefficients))

        array = np.arange(self.bbox.getArea(), dtype=np.float32).reshape(self.bbox.getDimensions())
        self.image = lsst.afw.image.ImageF(array)
        self.fields = [lsst.afw.math.ChebyshevBoundedField(self.bbox, coeffs) for _, coeffs in self.cases]
        self.product = lsst.afw.math.ProductBoundedField(self.fields)

    def tearDown(self):
        del self.bbox

    def testFillImageInterpolation(self):
        ctrl, coefficients = self.cases[-2]
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(10, 15),
                               lsst.geom.Extent2I(360, 350))
        field = lsst.afw.math.ChebyshevBoundedField(bbox, coefficients)
        image1 = lsst.afw.image.ImageF(bbox)
        image2 = lsst.afw.image.ImageF(bbox)
        image3 = lsst.afw.image.ImageF(bbox)
        image4 = lsst.afw.image.ImageF(bbox)
        field.fillImage(image1)
        field.fillImage(image2, xStep=3)
        field.fillImage(image3, yStep=4)
        field.fillImage(image4, xStep=3, yStep=4)
        self.assertFloatsAlmostEqual(image1.array, image2.array, rtol=1E-2, atol=1E-2)
        self.assertFloatsAlmostEqual(image1.array, image3.array, rtol=1.5E-2, atol=1.5E-2)
        self.assertFloatsAlmostEqual(image1.array, image4.array, rtol=2E-2, atol=2E-2)

    def testEvaluate(self):
        """Test the single-point evaluate method against explicitly-defined 1-d Chebyshevs
        (at the top of this file).
        """
        factor = 12.345
        boxD = lsst.geom.Box2D(self.bbox)
        # sx, sy: transform from self.bbox range to [-1, -1]
        sx = 2.0/boxD.getWidth()
        sy = 2.0/boxD.getHeight()
        nPoints = 50
        for ctrl, coefficients in self.cases:
            field = lsst.afw.math.ChebyshevBoundedField(
                self.bbox, coefficients)
            x = np.random.rand(nPoints)*boxD.getWidth() + boxD.getMinX()
            y = np.random.rand(nPoints)*boxD.getHeight() + boxD.getMinY()
            z1 = field.evaluate(x, y)
            tx = np.array([CHEBYSHEV_T[i](sx*x)
                           for i in range(coefficients.shape[1])])
            ty = np.array([CHEBYSHEV_T[i](sy*y)
                           for i in range(coefficients.shape[0])])
            self.assertEqual(tx.shape, (coefficients.shape[1], x.size))
            self.assertEqual(ty.shape, (coefficients.shape[0], y.size))
            z2 = np.array([np.dot(ty[:, i], np.dot(coefficients, tx[:, i]))
                           for i in range(nPoints)])
            self.assertFloatsAlmostEqual(z1, z2, rtol=1E-12)

            scaled = field*factor
            self.assertFloatsAlmostEqual(scaled.evaluate(x, y),
                                         factor*z2,
                                         rtol=factor*1E-13)
            self.assertFloatsEqual(
                scaled.getCoefficients(), factor*field.getCoefficients())

    def testProductEvaluate(self):
        """Test that ProductBoundedField.evaluate is equivalent to multiplying
        its nested BoundedFields.
        """
        zFlat1 = self.product.evaluate(self.xFlat, self.yFlat)
        zFlat2 = np.array([self.product.evaluate(x, y) for x, y in zip(self.xFlat, self.yFlat)])
        self.assertFloatsAlmostEqual(zFlat1, zFlat2)
        zFlat3 = np.ones(zFlat1.shape, dtype=float)
        for field in self.fields:
            zFlat3 *= field.evaluate(self.xFlat, self.yFlat)
        self.assertFloatsAlmostEqual(zFlat1, zFlat3)

    def testMultiplyImage(self):
        """Test multiplying an image in place.
        """
        _, coefficients = self.cases[-2]
        field = lsst.afw.math.ChebyshevBoundedField(self.image.getBBox(), coefficients)
        # multiplyImage() is in-place, so we have to make the expected result first.
        expect = multiply(self.image, field)
        field.multiplyImage(self.image)
        self.assertImagesAlmostEqual(self.image, expect)

    def testMultiplyMaskedImage(self):
        """Test multiplying a masked image in place.
        """
        _, coefficients = self.cases[-2]
        field = lsst.afw.math.ChebyshevBoundedField(self.image.getBBox(), coefficients)
        # multiplyImage() is in-place, so we have to make the expected result first.
        ones = lsst.afw.image.ImageF(self.image.getBBox())
        ones.array[:, :] = 1.0
        expect_masked_image = lsst.afw.image.MaskedImageF(
            multiply(self.image, field),
            None,
            multiply(multiply(ones, field), field)
        )
        masked_image = lsst.afw.image.MaskedImageF(self.image)
        masked_image.variance.array[:, :] = 1.0
        field.multiplyImage(masked_image)
        self.assertImagesAlmostEqual(masked_image.image, expect_masked_image.image)
        self.assertMaskedImagesAlmostEqual(masked_image, expect_masked_image)

    def testDivideImage(self):
        """Test dividing an image in place.
        """
        _, coefficients = self.cases[-2]
        field = lsst.afw.math.ChebyshevBoundedField(self.image.getBBox(), coefficients)
        # divideImage() is in-place, so we have to make the expected result first.
        expect = divide(self.image, field)
        field.divideImage(self.image)
        self.assertImagesAlmostEqual(self.image, expect)

    def testDivideMaskedImage(self):
        """Test dividing a masked image in place.
        """
        _, coefficients = self.cases[-2]
        field = lsst.afw.math.ChebyshevBoundedField(self.image.getBBox(), coefficients)
        # divideImage() is in-place, so we have to make the expected result first.
        ones = lsst.afw.image.ImageF(self.image.getBBox())
        ones.array[:, :] = 1.0
        expect_masked_image = lsst.afw.image.MaskedImageF(
            divide(self.image, field),
            None,
            divide(divide(ones, field), field)
        )
        masked_image = lsst.afw.image.MaskedImageF(self.image)
        masked_image.variance.array[:, :] = 1.0
        field.divideImage(masked_image)
        self.assertImagesAlmostEqual(masked_image.image, expect_masked_image.image)
        self.assertMaskedImagesAlmostEqual(masked_image, expect_masked_image)

    def testMultiplyImageRaisesUnequalBBox(self):
        """Multiplying an image with a different bbox should raise.
        """
        _, coefficients = self.cases[-2]
        field = lsst.afw.math.ChebyshevBoundedField(self.image.getBBox(), coefficients)
        subBox = lsst.geom.Box2I(lsst.geom.Point2I(0, 3), lsst.geom.Point2I(3, 4))
        subImage = self.image.subset(subBox)
        with self.assertRaises(RuntimeError):
            field.multiplyImage(subImage)

    def testMultiplyImageOverlapSubImage(self):
        """Multiplying a subimage with overlapOnly=true should only modify
        the subimage, when a subimage is passed in.
        """
        _, coefficients = self.cases[-2]
        field = lsst.afw.math.ChebyshevBoundedField(self.image.getBBox(), coefficients)
        subBox = lsst.geom.Box2I(lsst.geom.Point2I(0, 3), lsst.geom.Point2I(3, 4))
        subImage = self.image.subset(subBox)
        expect = self.image.clone()
        expect[subBox] = multiply(subImage, field)
        field.multiplyImage(subImage, overlapOnly=True)
        self.assertImagesAlmostEqual(self.image, expect)

    def testMultiplyImageOverlapSmallerBoundedField(self):
        """Multiplying a subimage with overlapOnly=true should only modify
        the subimage if the boundedField bbox is smaller than the image.

        This is checking for a bug where the bounded field was writing outside
        the overlap bbox.
        """
        _, coefficients = self.cases[-2]
        subBox = lsst.geom.Box2I(lsst.geom.Point2I(0, 3), lsst.geom.Point2I(3, 4))
        # The BF is only defined on the subBox, not the whole image bbox.
        field = lsst.afw.math.ChebyshevBoundedField(subBox, coefficients)
        subImage = self.image.subset(subBox)
        expect = self.image.clone()
        expect[subBox] = multiply(subImage, field)
        field.multiplyImage(self.image, overlapOnly=True)
        self.assertImagesAlmostEqual(self.image, expect)

    def _testIntegrateBox(self, bbox, coeffs, expect):
        field = lsst.afw.math.ChebyshevBoundedField(bbox, coeffs)
        self.assertFloatsAlmostEqual(field.integrate(), expect, rtol=1E-14)

    def testIntegrateTrivialBox(self):
        """Test integrating over a "trivial" [-1,1] box.

        NOTE: a "trivial" BBox can't be constructed exactly, given that Box2I
        is inclusive, but the [0,1] box has the same area (because it is
        actually (-0.5, 1.5) when converted to a Box2D), and the translation
        doesn't affect the integral.
        """
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                               lsst.geom.Point2I(1, 1))

        # 0th order polynomial
        coeffs = np.array([[5.0]])
        self._testIntegrateBox(bbox, coeffs, 4.0*coeffs[0, 0])

        # 1st order polynomial: odd orders drop out of integral
        coeffs = np.array([[5.0, 2.0], [3.0, 4.0]])
        self._testIntegrateBox(bbox, coeffs, 4.0*coeffs[0, 0])

        # 2nd order polynomial in x, 0th in y
        coeffs = np.array([[5.0, 0.0, 7.0]])
        self._testIntegrateBox(
            bbox, coeffs, 4.0*coeffs[0, 0] - (4.0/3.0)*coeffs[0, 2])

        # 2nd order polynomial in y, 0th in x
        coeffs = np.zeros((3, 3))
        coeffs[0, 0] = 5.0
        coeffs[2, 0] = 7.0
        self._testIntegrateBox(
            bbox, coeffs, 4.0*coeffs[0, 0] - (4.0/3.0)*coeffs[2, 0])

        # 2nd order polynomial in x and y, no cross-term
        coeffs = np.zeros((3, 3))
        coeffs[0, 0] = 5.0
        coeffs[1, 0] = 7.0
        coeffs[0, 2] = 3.0
        self._testIntegrateBox(bbox, coeffs,
                               4.0*coeffs[0, 0] - (4.0/3.0)*coeffs[2, 0] - (4.0/3.0)*coeffs[0, 2])

    def testIntegrateBox(self):
        r"""Test integrating over an "interesting" box.

        The values of these integrals were checked in Mathematica. The code
        block below can be pasted into Mathematica to re-do those calculations.

        ::

            f[x_, y_, n_, m_] := \!\(
                \*UnderoverscriptBox[\(\[Sum]\), \(i = 0\), \(n\)]\(
                \*UnderoverscriptBox[\(\[Sum]\), \(j = 0\), \(m\)]
                \*SubscriptBox[\(a\), \(i, j\)]*ChebyshevT[i, x]*ChebyshevT[j, y]\)\)
            integrate2dBox[n_, m_, x0_, x1_, y0_, y1_] := \!\(
                \*SubsuperscriptBox[\(\[Integral]\), \(y0\), \(y1\)]\(
                \*SubsuperscriptBox[\(\[Integral]\), \(x0\), \(x1\)]f[
                \*FractionBox[\(2  x - x0 - x1\), \(x1 - x0\)],
                \*FractionBox[\(2  y - y0 - y1\), \(y1 - y0\)], n,
                     m] \[DifferentialD]x \[DifferentialD]y\)\)
            integrate2dBox[0, 0, -2.5, 5.5, -3.5, 7.5]
            integrate2dBox[1, 0, -2.5, 5.5, -3.5, 7.5]
            integrate2dBox[0, 1, -2.5, 5.5, -3.5, 7.5]
            integrate2dBox[1, 1, -2.5, 5.5, -3.5, 7.5]
            integrate2dBox[1, 2, -2.5, 5.5, -3.5, 7.5]
            integrate2dBox[2, 2, -2.5, 5.5, -3.5, 7.5]
        """
        bbox = lsst.geom.Box2I(
            lsst.geom.Point2I(-2, -3), lsst.geom.Point2I(5, 7))

        # 0th order polynomial
        coeffs = np.array([[5.0]])
        self._testIntegrateBox(bbox, coeffs, 88.0*coeffs[0, 0])

        # 1st order polynomial: odd orders drop out of integral
        coeffs = np.array([[5.0, 2.0], [3.0, 4.0]])
        self._testIntegrateBox(bbox, coeffs, 88.0*coeffs[0, 0])

        # 2nd order polynomial in x, 0th in y
        coeffs = np.array([[5.0, 0.0, 7.0]])
        self._testIntegrateBox(
            bbox, coeffs, 88.0*coeffs[0, 0] - (88.0/3.0)*coeffs[0, 2])

        # 2nd order polynomial in y, 0th in x
        coeffs = np.zeros((3, 3))
        coeffs[0, 0] = 5.0
        coeffs[2, 0] = 7.0
        self._testIntegrateBox(
            bbox, coeffs, 88.0*coeffs[0, 0] - (88.0/3.0)*coeffs[2, 0])

        # 2nd order polynomial in x,y
        coeffs = np.zeros((3, 3))
        coeffs[2, 2] = 11.0
        self._testIntegrateBox(bbox, coeffs, (88.0/9.0)*coeffs[2, 2])

    def testMean(self):
        """The mean of the nth 1d Chebyshev (a_n*T_n(x)) on [-1,1] is
           0 for odd n
           a_n / (1-n^2) for even n

        Similarly, the mean of the (n,m)th 2d Chebyshev is the appropriate
        product of the above.
        """
        bbox = lsst.geom.Box2I(
            lsst.geom.Point2I(-2, -3), lsst.geom.Point2I(5, 7))

        coeffs = np.array([[5.0]])
        field = lsst.afw.math.ChebyshevBoundedField(bbox, coeffs)
        self.assertEqual(field.mean(), coeffs[0, 0])

        coeffs = np.array([[5.0, 0.0, 3.0]])
        field = lsst.afw.math.ChebyshevBoundedField(bbox, coeffs)
        self.assertEqual(field.mean(), coeffs[0, 0] - coeffs[0, 2]/3.0)

        # 2nd order polynomial in x,y
        coeffs = np.zeros((3, 3))
        coeffs[0, 0] = 7.0
        coeffs[1, 0] = 31.0
        coeffs[0, 2] = 13.0
        coeffs[2, 2] = 11.0
        field = lsst.afw.math.ChebyshevBoundedField(bbox, coeffs)
        self.assertFloatsAlmostEqual(
            field.mean(), coeffs[0, 0] - coeffs[0, 2]/3.0 + coeffs[2, 2]/9.0)

    def testImageFit(self):
        """Test that we can fit an image produced by a ChebyshevBoundedField and
        get the same coefficients back.
        """
        for ctrl, coefficients in self.cases:
            inField = lsst.afw.math.ChebyshevBoundedField(
                self.bbox, coefficients)
            for Image in (lsst.afw.image.ImageF, lsst.afw.image.ImageD):
                image = Image(self.bbox)
                inField.fillImage(image)
                outField = lsst.afw.math.ChebyshevBoundedField.fit(image, ctrl)
                self.assertFloatsAlmostEqual(
                    outField.getCoefficients(), coefficients, rtol=1E-6, atol=1E-7)

    def testArrayFit(self):
        """Test that we can fit 1-d arrays produced by a ChebyshevBoundedField and
        get the same coefficients back.
        """
        for ctrl, coefficients in self.cases:
            inField = lsst.afw.math.ChebyshevBoundedField(
                self.bbox, coefficients)
            for Image in (lsst.afw.image.ImageF, lsst.afw.image.ImageD):
                array = inField.evaluate(self.xFlat, self.yFlat)
                outField1 = lsst.afw.math.ChebyshevBoundedField.fit(self.bbox, self.xFlat, self.yFlat,
                                                                    array, ctrl)
                self.assertFloatsAlmostEqual(
                    outField1.getCoefficients(), coefficients, rtol=1E-6, atol=1E-7)
                weights = (1.0 + np.random.randn(array.size)**2)
                # Should get same results with different weights, since we still have no noise
                # and a model that can exactly reproduce the data.
                outField2 = lsst.afw.math.ChebyshevBoundedField.fit(self.bbox, self.xFlat, self.yFlat,
                                                                    array, weights, ctrl)
                self.assertFloatsAlmostEqual(
                    outField2.getCoefficients(), coefficients, rtol=1E-7, atol=1E-7)
                # If we make a matrix for the same points and use it to fit,
                # we get coefficients in unspecified order, but there are
                # (up to round-off error) the same as the nonzero entries of
                # the 2-d coefficients arrays.
                matrix = lsst.afw.math.ChebyshevBoundedField.makeFitMatrix(self.bbox, self.xFlat, self.yFlat,
                                                                           ctrl)
                coefficientsPacked, _, _, _ = np.linalg.lstsq(matrix, array)
                coefficientsPacked.sort()
                coefficientsFlat = coefficients.flatten()
                coefficientsFlat = coefficientsFlat[np.abs(coefficientsFlat) > 1E-7]
                coefficientsFlat.sort()
                self.assertFloatsAlmostEqual(coefficientsFlat, coefficientsPacked, rtol=1E-6, atol=1E-7)

    def testApproximate(self):
        """Test the approximate instantiation with the example of
        fitting a PixelAreaBoundedField to reasonable precision.
        """

        # This HSC-R band wcs was chosen arbitrarily from the edge of
        # field-of-view (ccd 4) for the w_2019_38 processing of RC2 as it
        # represents the typical use case of the approxBoundedField method.
        skyWcs = lsst.afw.geom.SkyWcs.readFits(os.path.join(testPath,
                                                            "data/jointcal_wcs-0034772-004.fits"))
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                               lsst.geom.Point2I(2047, 4175))

        pixelAreaField = lsst.afw.math.PixelAreaBoundedField(bbox, skyWcs,
                                                             unit=lsst.geom.arcseconds)
        approxField = lsst.afw.math.ChebyshevBoundedField.approximate(pixelAreaField)

        # Choose random points to test rather than a grid to ensure that
        # we are not using the same gridded points as used for the
        # approximation.
        np.random.seed(seed=1000)
        xTest = np.random.uniform(low=0.0, high=bbox.getMaxX(), size=10000)
        yTest = np.random.uniform(low=0.0, high=bbox.getMaxY(), size=10000)

        # The evaluation of approxField is ~80x faster than the
        # evaluation of pixelAreaField.
        expect = pixelAreaField.evaluate(xTest, yTest)
        result = approxField.evaluate(xTest, yTest)

        # The approximation is good to the 1e-7 level (absolute)
        self.assertFloatsAlmostEqual(result, expect, atol=1e-7)
        # and to the 1e-5 level (relative).  This is < 0.01 mmag.
        self.assertFloatsAlmostEqual(result, expect, rtol=1e-5)

    def testPersistence(self):
        """Test that we can round-trip a ChebyshevBoundedField through
        persistence.
        """
        boxD = lsst.geom.Box2D(self.bbox)
        nPoints = 50
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            for ctrl, coefficients in self.cases:
                inField = lsst.afw.math.ChebyshevBoundedField(
                    self.bbox, coefficients)
                inField.writeFits(filename)
                outField = lsst.afw.math.ChebyshevBoundedField.readFits(filename)
                self.assertEqual(inField.getBBox(), outField.getBBox())
                self.assertFloatsAlmostEqual(
                    inField.getCoefficients(), outField.getCoefficients())
                x = np.random.rand(nPoints)*boxD.getWidth() + boxD.getMinX()
                y = np.random.rand(nPoints)*boxD.getHeight() + boxD.getMinY()
                z1 = inField.evaluate(x, y)
                z2 = inField.evaluate(x, y)
                self.assertFloatsAlmostEqual(z1, z2, rtol=1E-13)

            # test with an empty bbox
            inField = lsst.afw.math.ChebyshevBoundedField(lsst.geom.Box2I(),
                                                          np.array([[1.0, 2.0], [3.0, 4.0]]))
            inField.writeFits(filename)
            outField = lsst.afw.math.ChebyshevBoundedField.readFits(filename)
            self.assertEqual(inField.getBBox(), outField.getBBox())

    def testProductPersistence(self):
        """Test that we can round-trip a ProductBoundedField through
        persistence.
        """
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            self.product.writeFits(filename)
            out = lsst.afw.math.ProductBoundedField.readFits(filename)
            self.assertEqual(self.product, out)

    def testTruncate(self):
        """Test that truncate() works as expected.
        """
        for ctrl, coefficients in self.cases:
            field1 = lsst.afw.math.ChebyshevBoundedField(
                self.bbox, coefficients)
            field2 = field1.truncate(ctrl)
            self.assertFloatsAlmostEqual(
                field1.getCoefficients(), field2.getCoefficients())
            self.assertEqual(field1.getBBox(), field2.getBBox())
            config3 = lsst.afw.math.ChebyshevBoundedField.ConfigClass()
            config3.readControl(ctrl)
            if ctrl.orderX > 0:
                config3.orderX -= 1
            if ctrl.orderY > 0:
                config3.orderY -= 1
            field3 = field1.truncate(config3.makeControl())
            for i in range(config3.orderY + 1):
                for j in range(config3.orderX + 1):
                    if config3.triangular and i + j > max(config3.orderX, config3.orderY):
                        self.assertEqual(field3.getCoefficients()[i, j], 0.0)
                    else:
                        self.assertEqual(field3.getCoefficients()[i, j],
                                         field1.getCoefficients()[i, j])

    def testEquality(self):
        for ctrl, coefficients in self.cases:
            field1 = lsst.afw.math.ChebyshevBoundedField(self.bbox, coefficients)
            field2 = lsst.afw.math.ChebyshevBoundedField(self.bbox, coefficients)
            self.assertEqual(field1, field2, msg=coefficients)

        # same coefficients, instantiated from different arrays
        field1 = lsst.afw.math.ChebyshevBoundedField(self.bbox, np.array([[1.0]]))
        field2 = lsst.afw.math.ChebyshevBoundedField(self.bbox, np.array([[1.0]]))
        self.assertEqual(field1, field2)
        field1 = lsst.afw.math.ChebyshevBoundedField(self.bbox, np.array([[1.0, 2.0], [3., 4.]]))
        field2 = lsst.afw.math.ChebyshevBoundedField(self.bbox, np.array([[1.0, 2.0], [3., 4.]]))
        self.assertEqual(field1, field2)

        # different coefficient(s)
        field1 = lsst.afw.math.ChebyshevBoundedField(self.bbox, np.array([[1.0]]))
        field2 = lsst.afw.math.ChebyshevBoundedField(self.bbox, np.array([[2.0]]))
        self.assertNotEqual(field1, field2)
        field1 = lsst.afw.math.ChebyshevBoundedField(self.bbox, np.array([[1.0, 0.0]]))
        field2 = lsst.afw.math.ChebyshevBoundedField(self.bbox, np.array([[1.0], [0.0]]))
        self.assertNotEqual(field1, field2)

        # different bbox
        bbox1 = lsst.geom.Box2I(lsst.geom.Point2I(-10, -10), lsst.geom.Point2I(5, 5))
        bbox2 = lsst.geom.Box2I(lsst.geom.Point2I(-5, -5), lsst.geom.Point2I(5, 5))
        field1 = lsst.afw.math.ChebyshevBoundedField(bbox1, np.array([[1.0]]))
        field2 = lsst.afw.math.ChebyshevBoundedField(bbox2, np.array([[1.0]]))
        self.assertNotEqual(field1, field2)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
