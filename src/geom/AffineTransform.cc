#include "Eigen/LU"

#include <iomanip>

#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/pex/exceptions/Runtime.h"

namespace geom = lsst::afw::geom;

/** 
 * @brief construct a vector from non-zero elements of the transform
 *
 * The other of the vector is [XX, YX, XY, YY, X, Y]
 */
geom::AffineTransform::ParameterVector geom::AffineTransform::getVector() const {
    ParameterVector r;
    r << _eigenTransform(0,0),
         _eigenTransform(1,0), 
         _eigenTransform(0,1), 
         _eigenTransform(1,1), 
         _eigenTransform(0,2), 
         _eigenTransform(1,2);
    return r;
}

/**
 * @brief Return the inverse transform
 *
 * @throw lsst::pex::exceptions::RuntimeException if this is not invertible
 */
geom::AffineTransform geom::AffineTransform::invert() const {
    Eigen::LU<Eigen::Matrix2d> lu(_eigenTransform.linear());
    if (!lu.isInvertible()) {
        throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeErrorException,
                "Matrix cannot be inverted"
        );
    }
    Eigen::Matrix2d inv = lu.inverse();
    EigenPoint p = -inv*_eigenTransform.translation();
    return AffineTransform(inv, lsst::afw::geom::ExtentD(p));
}

/**
 * @brief Take the derivative of (*this)(input) w.r.t the transform elements
 */
geom::AffineTransform::TransformDerivativeMatrix geom::AffineTransform::dTransform(
    PointD const & input
) const {
    TransformDerivativeMatrix r = TransformDerivativeMatrix::Zero();
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
    TransformDerivativeMatrix r = TransformDerivativeMatrix::Zero();
    r(0,XX) = input.getX();
    r(0,XY) = input.getY();
    r(1,YX) = input.getX();
    r(1,YY) = input.getY();
    return r;
}

geom::AffineTransform const & geom::AffineTransform::operator =(
    ParameterVector const & vector
) {
    _eigenTransform.matrix().block<2, 3>(0,0) << 
        vector[0], vector[2], vector[4], 
        vector[1], vector[3], vector[5];
    return *this; 
}
geom::AffineTransform const & geom::AffineTransform::operator =(
    geom::AffineTransform::EigenTransform const & matrix
) {
    _eigenTransform = matrix;
    return *this;
}

geom::AffineTransform const & geom::AffineTransform::operator =(
    geom::AffineTransform const & transform
) {
    _eigenTransform = transform.getEigenTransform();
    return *this;
}

std::ostream& geom::operator<<(std::ostream& os, geom::AffineTransform const & transform) {
    std::ios::fmtflags flags = os.flags();
    geom::AffineTransform::Matrix const & matrix = transform.getMatrix();
    int prec = os.precision(7);
    os.setf(std::ios::fixed);
    os << "AffineTransform([(" << std::setw(10) << matrix(0,0) << "," << std::setw(10) << matrix(0,1) 
       << "," << std::setw(10) << matrix(0,2) << "),\n";
    os << "                 (" << std::setw(10) << matrix(1,0) << "," << std::setw(10) << matrix(1,1)
       << "," << std::setw(10) << matrix(1,2) << "),\n";
    os << "                 (" << std::setw(10) << matrix(2,0) << "," << std::setw(10) << matrix(2,1)
       << "," << std::setw(10) << matrix(2,2) << ")])";
    os.precision(prec);
    os.flags(flags);
    return os;
}
