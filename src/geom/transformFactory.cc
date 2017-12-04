// -*- LSST-C++ -*-
/*
 * LSST Data Management System
 * See COPYRIGHT file at the top of the source tree.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program. If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#include <sstream>
#include <cmath>

#include "astshim.h"

#include "Eigen/Core"

#include "lsst/afw/geom/Endpoint.h"
#include "lsst/afw/geom/transformFactory.h"
#include "lsst/pex/exceptions.h"

#include "ndarray.h"
#include "ndarray/eigen.h"

namespace lsst {
namespace afw {
namespace geom {

namespace {
/*
 * Print a vector to a stream.
 *
 * The exact details of the representation are unspecified and subject to
 * change, but the following may be regarded as typical:
 *
 *     [1.0, -3.560, 42.0]
 *
 * @tparam T the element type. Must support stream output.
 */
template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> const &v) {
    os << '[';
    bool first = true;
    for (T element : v) {
        if (first) {
            first = false;
        } else {
            os << ", ";
        }
        os << element;
    }
    os << ']';
    return os;
}

/*
 * Convert a Matrix to the equivalent ndarray.
 *
 * @param matrix The matrix to convert.
 * @returns an ndarray containing a copy of the data in `matrix`
 *
 * @note Returning a copy is safer that returning a view because ndarray cannot share
 * management of the memory with Eigen. So as long as the matrix is small,
 * making a copy is preferred even if a view would suffice.
 */
template <typename Derived>
ndarray::Array<typename Eigen::MatrixBase<Derived>::Scalar, 2, 2> toNdArray(
        Eigen::MatrixBase<Derived> const &matrix) {
    ndarray::Array<typename Eigen::MatrixBase<Derived>::Scalar, 2, 2> array =
            ndarray::allocate(ndarray::makeVector(matrix.rows(), matrix.cols()));
    array.asEigen() = matrix;
    return array;
}

/*
 * Tests whether polynomial coefficients match the expected format.
 *
 * @param coeffs radial polynomial coefficients.
 * @returns `true` if either `coeffs.size()` = 0, or `coeffs.size()` > 1,
 *          `coeffs[0]` = 0, and `coeffs[1]` &ne; 0. `false` otherwise.
 */
bool areRadialCoefficients(std::vector<double> const &coeffs) noexcept {
    if (coeffs.empty()) {
        return true;
    } else {
        return coeffs.size() > 1 && coeffs[0] == 0.0 && coeffs[1] != 0.0;
    }
}

/*
 * Make a one-dimensional polynomial distortion.
 *
 * The Mapping computes a scalar function
 * @f[ f(x) = \sum_{i=1}^{N} \mathrm{coeffs[i]} \ x^i @f]
 *
 * @param coeffs radial polynomial coefficients. Must have `size` > 1,
 *               `coeffs[0]` = 0, and `coeffs[1]` &ne; 0.
 * @returns the function represented by `coeffs`. The Mapping shall have an
 *          inverse, which may be approximate.
 *
 * @exceptsafe Provides basic exception safety.
 *
 * @warning Input to this function is not validated.
 */
ast::PolyMap makeOneDDistortion(std::vector<double> const &coeffs) {
    int const nCoeffs = coeffs.size() - 1;  // ignore coeffs[0]
    ndarray::Array<double, 2, 2> const polyCoeffs = ndarray::allocate(ndarray::makeVector(nCoeffs, 3));
    for (size_t i = 1; i < coeffs.size(); ++i) {
        polyCoeffs[i - 1][0] = coeffs[i];
        polyCoeffs[i - 1][1] = 1;
        polyCoeffs[i - 1][2] = i;
    }

    return ast::PolyMap(polyCoeffs, 1, "IterInverse=1, TolInverse=1e-8, NIterInverse=20");
}

}  // namespace

AffineTransform linearizeTransform(TransformPoint2ToPoint2 const &original, Point2D const &inPoint) {
    auto outPoint = original.applyForward(inPoint);
    Eigen::Matrix2d jacobian = original.getJacobian(inPoint);
    for (int i = 0; i < 2; ++i) {
        if (!std::isfinite(outPoint[i])) {
            std::ostringstream buffer;
            buffer << "Transform ill-defined: " << inPoint << " -> " << outPoint;
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
        }
    }
    if (!jacobian.allFinite()) {
        std::ostringstream buffer;
        buffer << "Transform not continuous at " << inPoint << ": J = " << jacobian;
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
    }

    // y(x) = J (x - x0) + y0 = J x + (y0 - J x0)
    auto offset = outPoint.asEigen() - jacobian * inPoint.asEigen();
    return AffineTransform(jacobian, offset);
}

std::shared_ptr<TransformPoint2ToPoint2> makeTransform(AffineTransform const &affine) {
    auto const offset = Point2D(affine.getTranslation());
    auto const jacobian = affine.getLinear().getMatrix();

    Point2Endpoint toEndpoint;
    auto const map = ast::MatrixMap(toNdArray(jacobian))
                             .then(ast::ShiftMap(toEndpoint.dataFromPoint(offset))).simplify();
    return std::make_shared<TransformPoint2ToPoint2>(*map);
}

std::shared_ptr<TransformPoint2ToPoint2> makeRadialTransform(std::vector<double> const &coeffs) {
    if (!areRadialCoefficients(coeffs)) {
        std::ostringstream buffer;
        buffer << "Invalid coefficient vector: " << coeffs;
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
    }

    if (coeffs.empty()) {
        return std::make_shared<TransformPoint2ToPoint2>(ast::UnitMap(2));
    } else {
        // distortion is a radial polynomial with center at focal plane center;
        // the polynomial has an iterative inverse
        std::vector<double> center = {0.0, 0.0};
        ast::PolyMap const distortion = makeOneDDistortion(coeffs);
        return std::make_shared<TransformPoint2ToPoint2>(*ast::makeRadialMapping(center, distortion));
    }
}

std::shared_ptr<TransformPoint2ToPoint2> makeRadialTransform(std::vector<double> const &forwardCoeffs,
                                                             std::vector<double> const &inverseCoeffs) {
    if (forwardCoeffs.empty() != inverseCoeffs.empty()) {
        throw LSST_EXCEPT(
                pex::exceptions::InvalidParameterError,
                "makeRadialTransform must have either both empty or both non-empty coefficient vectors.");
    }
    if (forwardCoeffs.empty()) {
        // no forward or inverse coefficients, so no distortion
        return std::make_shared<TransformPoint2ToPoint2>(ast::UnitMap(2));
    }

    if (!areRadialCoefficients(forwardCoeffs)) {
        std::ostringstream buffer;
        buffer << "Invalid forward coefficient vector: " << forwardCoeffs;
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
    }
    if (!areRadialCoefficients(inverseCoeffs)) {
        std::ostringstream buffer;
        buffer << "Invalid inverse coefficient vector: " << inverseCoeffs;
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
    }
    // distortion is a 1-d radial polynomial centered at focal plane center;
    // the polynomial has coefficients specified for both the forward and inverse directions
    std::vector<double> center = {0.0, 0.0};
    ast::PolyMap const forward = makeOneDDistortion(forwardCoeffs);
    auto inverse = makeOneDDistortion(inverseCoeffs).getInverse();
    auto distortion = ast::TranMap(forward, *inverse);
    return std::make_shared<TransformPoint2ToPoint2>(*ast::makeRadialMapping(center, distortion));
}

std::shared_ptr<TransformPoint2ToPoint2> makeIdentityTransform() {
    return std::make_shared<TransformPoint2ToPoint2>(ast::UnitMap(2));
}

}  // namespace geom
}  // namespace afw
}  // namespace lsst
