#include "Eigen/LU"

#include "lsst/afw/math/AffineTransform.h"
#include "lsst/pex/exceptions/Runtime.h"

namespace math = lsst::afw::math;

/** 
 * @brief construct a vector from non-zero elements of the transform
 *
 * The other of the vector is [XX, YX, XY, YY, X, Y]
 */
Eigen::Vector6d math::AffineTransform::getVector() const {
    Eigen::Vector6d r;

    r << _matrix(0,0),
         _matrix(1,0), 
         _matrix(0,1), 
         _matrix(1,1), 
         _matrix(0,2), 
         _matrix(1,2);
    return r;
}

/**
 * @brief Return the inverse transform
 *
 * @throw lsst::pex::exceptions::RuntimeException if this is not invertible
 */
math::AffineTransform* math::AffineTransform::invert() const {
    Eigen::LU<Eigen::Matrix2d> lu(_matrix.linear());
    if (!lu.isInvertible()) {
        throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeErrorException,
                "Matrix cannot be inverted"
        );
    }
    Eigen::Matrix2d inv = lu.inverse();
    EigenPoint p = -inv*_matrix.translation();
    return new AffineTransform(inv, lsst::afw::image::PointD(p.x(), p.y()));
}

/**
 * @brief Take the derivative of (*this)(input) w.r.t the transform elements
 */
Eigen::Matrix<double,2,6> math::AffineTransform::d(
        PointD const & input
) const {
    Eigen::Matrix<double,2,6> r;
    r(0,XX) = input.getX();
    r(0,XY) = input.getY();
    r(0,X) = 1.0;
    r(1,YX) = input.getX();
    r(1,YY) = input.getY();
    r(1,Y) = 1.0;
    return r;
}

math::AffineTransform const & math::AffineTransform::operator =(
    Eigen::Vector6d const & vector
) {
    _matrix.matrix().block<2, 3>(0,0) << 
            vector[0], vector[2], vector[4], 
            vector[1], vector[3], vector[5];
    return *this; 
}
math::AffineTransform const & math::AffineTransform::operator =(
    math::AffineTransform::TransformMatrix const & matrix
) {
    _matrix = matrix;
    return *this;
}

math::AffineTransform const & math::AffineTransform::operator =(
    math::AffineTransform const & transform
) {
    _matrix = transform.matrix();
    return *this;
}
