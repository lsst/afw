// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2013 LSST Corporation.
 *
 * This product includes so@ftware developed by the
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

#ifndef LSST_AFW_GEOM_AFFINE_TRANSFORM_H
#define LSST_AFW_GEOM_AFFINE_TRANSFORM_H

#include <boost/shared_ptr.hpp>
#include "Eigen/Core"
#include <iostream>

#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/LinearTransform.h"

namespace lsst {
namespace afw {
namespace geom {

/**
 *  \brief An affine coordinate transformation consisting of a linear transformation and an offset.
 *
 *  The transform is represented by a matrix @f$ \mathbf{M} @f$ such that
 *  \f[
 *     \left[\begin{array}{ c }
 *     x_f \\
 *     y_f \\
 *     1
 *     \end{array}\right]
 *     =
 *     \mathbf{M}
 *     \left[\begin{array}{ c }
 *     x_i \\
 *     y_i \\
 *     1
 *     \end{array}\right]
 *  \f]
 *  where @f$(x_i,y_i)@f$ are the input coordinates and @f$(x_f,y_f)@f$ are the output coordinates.
 *
 *  If @f$ x_f(x_i,y_i) @f$ and @f$ y_f(x_i,y_i) @f$ are continuous differentiable functions, then
 *  \f[
 *     \mathbf{M} = \left[\begin{array}{ c c c }
 *     \displaystyle\frac{\partial x_f}{\partial x_i} &
 *     \displaystyle\frac{\partial x_f}{\partial y_i} &
 *     x_f \\
 *     \displaystyle\frac{\partial y_f}{\partial x_i} &
 *     \displaystyle\frac{\partial y_f}{\partial y_i} &
 *     y_f \\
 *     \displaystyle 0 & \displaystyle 0 & \displaystyle 1
 *     \end{array}\right]
 *  \f]
 *  evaluated at @f$(x_i,y_i)@f$.
 *
 *  The 2x2 upper left corner of @f$ \mathbf{M} @f$ (the LinearTransform part) is simply the
 *  Jacobian of the mapping between @f$(x_i,y_i)@f$ and @f$(x_f,y_f)@f$.
 */
class AffineTransform {
public:
    /// Enum to provide indices for matrix elements when we have to flatten them (e.g. for derivatives)
    enum Parameters {XX=0,YX=1,XY=2,YY=3,X=4,Y=5};

    /// Matrix type used to define the transform itself.
    typedef Eigen::Matrix3d Matrix;

    /// Type used when flattening matrix elements into a vector.
    typedef Eigen::Matrix<double,6,1> ParameterVector;

    /// Type used when computing the derivative of a transformed point with respect to the transform.
    typedef Eigen::Matrix<double,2,6> TransformDerivativeMatrix;

    /// Construct an identity AffineTransform.
    AffineTransform() : _linear(), _translation() {}

    /// Construct an AffineTransform from a 3x3 matrix.
    explicit AffineTransform(Eigen::Matrix3d const & matrix)
      : _linear(matrix.block<2,2>(0,0)),
        _translation(matrix.block<2,1>(0,2))
    {}

    /// Construct an AffineTransform with no translation from a 2x2 matrix.
    explicit AffineTransform(Eigen::Matrix2d const & linear)
      : _linear(linear), _translation() {}

    /// Construct a translation-only AffineTransform from a vector.
    explicit AffineTransform(Eigen::Vector2d const & translation)
      : _linear(), _translation(translation) {}

    /// Construct an AffineTransform from a 2x2 matrix and vector.
    explicit AffineTransform(
        Eigen::Matrix2d const & linear, Eigen::Vector2d const & translation
    ) : _linear(linear), _translation(translation) {}

    /// Construct an AffineTransform from a LinearTransform.
    explicit AffineTransform(LinearTransform const & linear)
      : _linear(linear), _translation() {}

    /// Construct a translation-only AffineTransform from an Extent2D.
    explicit AffineTransform(Extent2D const & translation)
      : _linear(), _translation(translation) {}

    /// Construct an AffineTransform from a LinearTransform and Extent2D.
    explicit AffineTransform(
        LinearTransform const & linear, Extent2D const & translation
    ) : _linear(linear), _translation(translation) {}

    /// Invert the transform (just matrix inversion).
    AffineTransform const invert() const;

    /// Whether the transform is a no-op.
    bool isIdentity() const { return getMatrix().isIdentity(); }

    /**
     * @brief Transform a Point object.
     *
     * The result is affected by the translation parameters of the transform
     */
    Point2D operator()(Point2D const &p) const {
        return Point2D(_linear(p) + _translation);
    }

    /**
     * @brief Transform an Extent object.
     *
     * The result is unaffected by the translation parameters of the transform
     */
    Extent2D operator()(Extent2D const &p) const {
        return Extent2D(_linear(p));
    }

    //@{
    /// Return an internal reference to the translation part of the transform as an Extent2D
    Extent2D const & getTranslation() const { return _translation; }
    Extent2D & getTranslation() { return _translation; }
    //@}

    //@{
    /// Return an internal reference to the upper 2x2 part of the transform as a LinearTransform
    LinearTransform const & getLinear() const { return _linear; }
    LinearTransform & getLinear() { return _linear; }
    //@}

    /// Return the transform matrix as an Eigen object.
    Matrix const getMatrix() const;

    /// Return the matrix elements as a flattened vector, ordered according to the Parameters enum.
    ParameterVector const getParameterVector() const;

    /// Set the matrix elements from a flattened vector, ordered according to the Parameters enum.
    void setParameterVector(ParameterVector const & vector);

    //@{
    /// Return an element of the transform matrix, as indexed by the Parameters enum.
    double & operator[](int i) {
        return (i < 4) ? _linear[i] : _translation[i - 4];
    }
    double operator[](int i) const {
        return (i < 4) ? _linear[i] : _translation[i - 4];
    }
    //@}

    /// Composition of transforms (just matrix multiplication).
    AffineTransform operator*(AffineTransform const & other) const {
        return AffineTransform(
            getLinear()*other.getLinear(),
            getLinear()(other.getTranslation()) + getTranslation()
        );
    }

    /// Standard assignment.
    AffineTransform & operator=(AffineTransform const & other) {
        _linear = other._linear;
        _translation = other._translation;
        return *this;
    }

    /**
     *  @brief Construct a new AffineTransform that represents a uniform scaling.
     *
     *  @return An AffineTransform with matrix
     *  @f[
     *     \left[\begin{array}{ c c c }
     *     s & 0 & 0 \\
     *     0 & s & 0 \\
     *     0 & 0 & 1 \\
     *     \end{array}\right]
     *  @f]
     */
    static AffineTransform makeScaling(double s) {
        return AffineTransform(LinearTransform::makeScaling(s));
    }

    /**
     *  @brief Construct a new AffineTransform that represents a scaling along the coordinate axes.
     *
     *  @return An AffineTransform with matrix
     *  @f[
     *     \left[\begin{array}{ c c c }
     *     s & 0 & 0 \\
     *     0 & t & 0 \\
     *     0 & 0 & 1 \\
     *     \end{array}\right]
     *  @f]
     */
    static AffineTransform makeScaling(double s, double t) {
        return AffineTransform(LinearTransform::makeScaling(s, t));
    }
    /**
     *  @brief Construct a new AffineTransform that represents a CCW rotation.
     *
     *  @return An AffineTransform with matrix
     *  @f[
     *     \left[\begin{array}{ c c c }
     *     \cos t & -\sin t & 0 \\
     *     \sin t & \cos t & 0  \\
     *     0 & 0 & 1 \\
     *     \end{array}\right]
     *  @f]
     */
    static AffineTransform makeRotation(Angle t) {
        return AffineTransform(LinearTransform::makeRotation(t));
    }

    /**
     *  @brief Construct a new AffineTransform that represents a pure translation.
     *
     *  @return An AffineTransform with matrix
     *  @f[
     *     \left[\begin{array}{ c c c }
     *     0 & 0 & translation.getX() \\
     *     0 & 0 & translation.getY() \\
     *     0 & 0 & 1 \\
     *     \end{array}\right]
     *  @f]
     */
    static AffineTransform makeTranslation(Extent2D translation) {
        return AffineTransform(translation);
    }

    //@{
    /// Return the derivative of the transformed point with respect to the transform matrix elements.
    TransformDerivativeMatrix dTransform(Point2D const & input) const;
    TransformDerivativeMatrix dTransform(Extent2D const & input) const;
    //@}

private:
    LinearTransform _linear;
    Extent2D _translation;
};

std::ostream & operator<<(std::ostream & os, lsst::afw::geom::AffineTransform const & transform);

/// Returns the unique AffineTransform A such that A(p_i)=q_i for i=1,2,3
AffineTransform makeAffineTransformFromTriple(Point2D const &p1, Point2D const &p2, Point2D const &p3,
                                              Point2D const &q1, Point2D const &q2, Point2D const &q3);

}}} // namespace lsst::afw::geom

#endif // !LSST_AFW_GEOM_AFFINE_TRANSFORM_H
