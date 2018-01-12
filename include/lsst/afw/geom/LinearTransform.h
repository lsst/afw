/*
 * LSST Data Management System
 * Copyright 2008-2014 LSST Corporation.
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#ifndef LSST_AFW_GEOM_LINEAR_TRANSFORM_H
#define LSST_AFW_GEOM_LINEAR_TRANSFORM_H

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "lsst/pex/exceptions/Runtime.h"

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Angle.h"

namespace lsst {
namespace afw {
namespace geom {

LSST_EXCEPTION_TYPE(SingularTransformException, lsst::pex::exceptions::RuntimeError,
                    lsst::afw::geom::SingularTransformException)

/**
 *  A 2D linear coordinate transformation.
 *
 *  The transform is represented by a matrix @f$ \mathbf{M} @f$ such that
 *  @f[
 *     \left[\begin{array}{ c }
 *     x_f \\
 *     y_f
 *     \end{array}\right]
 *     =
 *     \mathbf{M}
 *     \left[\begin{array}{ c }
 *     x_i \\
 *     y_i
 *     \end{array}\right]
 *  @f]
 *  where @f$(x_i,y_i)@f$ are the input coordinates and @f$(x_f,y_f)@f$ are
 *  the output coordinates.
 *
 *  If @f$ x_f(x_i,y_i) @f$ and @f$ y_f(x_i,y_i) @f$ are continuous
 *  differentiable functions, then
 *  @f[
 *     \mathbf{M} = \left[\begin{array}{ c c }
 *     \displaystyle\frac{\partial x_f}{\partial x_i} &
 *     \displaystyle\frac{\partial x_f}{\partial y_i} \\
 *     \displaystyle\frac{\partial y_f}{\partial x_i} &
 *     \displaystyle\frac{\partial y_f}{\partial y_i}
 *     \end{array}\right]
 *  @f]
 *  evaluated at @f$(x_i,y_i)@f$.
 */
class LinearTransform {
public:
    enum Parameters { XX = 0, YX = 1, XY = 2, YY = 3 };

    typedef Eigen::Matrix<double, 4, 1> ParameterVector;
    typedef Eigen::Matrix<double, 2, 4> TransformDerivativeMatrix;
    typedef Eigen::Matrix<double, 4, 4> ProductDerivativeMatrix;

    typedef Eigen::Matrix<double, 2, 2, Eigen::DontAlign> Matrix;

    /** Construct an empty (identity) LinearTransform. */
    LinearTransform() : _matrix(Matrix::Identity()) {}

    /** Construct an LinearTransform from an Eigen::Matrix. */
    explicit LinearTransform(Matrix const& matrix) : _matrix(matrix) {}

    LinearTransform(LinearTransform const&) = default;
    LinearTransform(LinearTransform&&) = default;
    ~LinearTransform() = default;

    LinearTransform operator*(LinearTransform const& other) const {
        return LinearTransform(getMatrix() * other.getMatrix());
    }

    static LinearTransform makeScaling(double s) {
        return LinearTransform((Matrix() << s, 0.0, 0.0, s).finished());
    }

    static LinearTransform makeScaling(double s, double t) {
        return LinearTransform((Matrix() << s, 0.0, 0.0, t).finished());
    }

    static LinearTransform makeRotation(Angle t) {
        return LinearTransform(Matrix(Eigen::Rotation2D<double>(t.asRadians())));
    }

    LinearTransform& operator=(LinearTransform const&) = default;
    LinearTransform& operator=(LinearTransform&&) = default;

    LinearTransform& operator+=(LinearTransform const& other) {
        _matrix += other._matrix;
        return *this;
    }

    LinearTransform operator+(LinearTransform const& other) {
        LinearTransform tmp(*this);
        tmp += other;
        return tmp;
    }

    LinearTransform& operator-=(LinearTransform const& other) {
        _matrix -= other._matrix;
        return *this;
    }

    LinearTransform operator-(LinearTransform const& other) {
        LinearTransform tmp(*this);
        tmp -= other;
        return tmp;
    }

    /**
     * Return the transform matrix elements as a parameter vector
     *
     * The elements will be ordered XX, YX, XY, YY
     */
    ParameterVector const getParameterVector() const;
    /**
     * Set the transform matrix elements from a parameter vector
     *
     * The parameter vector is ordered XX, YX, XY, YY
     */
    void setParameterVector(ParameterVector const& vector);

    Matrix const& getMatrix() const { return _matrix; }
    Matrix& getMatrix() { return _matrix; }

    double& operator[](int i) { return _matrix(i % 2, i / 2); }
    double const& operator[](int i) const { return const_cast<Matrix&>(_matrix)(i % 2, i / 2); }

    /**
     * Return the inverse transform.
     *
     * @throws lsst::afw::geom::SingularTransformException
     */
    LinearTransform const invert() const;

    /**
     * Return the determinant of the 2x2 matrix
     */
    double computeDeterminant() const;

    /** Whether the transform is a no-op. */
    bool isIdentity() const { return getMatrix().isIdentity(); }

    /**
     *  Transform a Point2D object.
     *
     *  This operation is equivalent to applying the LinearTransform to an
     *  lsst::afw::geom::Extent
     */
    Point2D operator()(Point2D const& p) const { return Point2D(getMatrix() * p.asEigen()); }

    /**
     *  Transform a Extent2D object.
     *
     *  This operation is equivalent to applying the LinearTransform to an
     *  lsst::afw::geom::Point
     */
    Extent2D operator()(Extent2D const& p) const { return Extent2D(getMatrix() * p.asEigen()); }

    /**
     * Derivative of (*this)(input) with respect to the transform elements (for Point).
     */
    TransformDerivativeMatrix dTransform(Point2D const& input) const;

    /// Derivative of (*this)(input) with respect to the transform elements (for Extent);
    TransformDerivativeMatrix dTransform(Extent2D const& input) const { return dTransform(Point2D(input)); }

private:
    Matrix _matrix;
};

std::ostream& operator<<(std::ostream& os, lsst::afw::geom::LinearTransform const& t);
}
}
}  // namespace lsst::afw::geom

#endif  // !LSST_AFW_GEOM_LINEAR_TRANSFORM_H
