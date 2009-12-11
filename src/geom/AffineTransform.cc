#include "Eigen/LU"

#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/pex/exceptions/Runtime.h"

namespace math = lsst::afw::geom;

/** 
 * @brief construct a vector from non-zero elements of the transform
 *
 * The other of the vector is [XX, YX, XY, YY, X, Y]
 */
geom::AffineTransform::ParameterVector geom::AffineTransform::getVector() const {
    ParameterVector r;

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
geom::AffineTransform geom::AffineTransform::invert() const {
    Eigen::LU<Eigen::Matrix2d> lu(_matrix.linear());
    if (!lu.isInvertible()) {
        throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeErrorException,
                "Matrix cannot be inverted"
        );
    }
    Eigen::Matrix2d inv = lu.inverse();
    EigenPoint p = -inv*_matrix.translation();
    return AffineTransform(inv, lsst::afw::image::PointD(p.x(), p.y()));
}

/**
 * @brief Take the derivative of (*this)(input) w.r.t the transform elements
 */
geom::AffineTransform::TransformDerivativeMatrix geom::AffineTransform::dTransform(
    PointD const & input
) const {
    TransformDerivativeMatrix r = transformDerivativeMatrix::Zero();
    r(0,XX) = input.getX();
    r(0,XY) = input.getY();
    r(0,X) = 1.0;
    r(1,YX) = input.getX();
    r(1,YY) = input.getY();
    r(1,Y) = 1.0;
    return r;
}

/**
 * @brief Take the derivative of (*this)(input) w.r.t the transform elements
 */
geom::AffineTransform::TransformDerivativeMatrix geom::AffineTransform::dTransform(
    ExtentD const & input
) const {
    TransformDerivativeMatrix r = transformDerivativeMatrix::Zero();
    r(0,XX) = input.getX();
    r(0,XY) = input.getY();
    r(1,YX) = input.getX();
    r(1,YY) = input.getY();
    return r;
}

geom::AffineTransform const & geom::AffineTransform::operator =(
    ParameterVector const & vector
) {
    _matrix.matrix().block<2, 3>(0,0) << 
            vector[0], vector[2], vector[4], 
            vector[1], vector[3], vector[5];
    return *this; 
}
geom::AffineTransform const & geom::AffineTransform::operator =(
    geom::AffineTransform::TransformMatrix const & matrix
) {
    _matrix = matrix;
    return *this;
}

geom::AffineTransform const & geom::AffineTransform::operator =(
    geom::AffineTransform const & transform
) {
    _matrix = transform.matrix();
    return *this;
}
