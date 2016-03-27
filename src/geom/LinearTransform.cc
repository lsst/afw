/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include "lsst/afw/geom/LinearTransform.h"

#include "Eigen/LU"

#include <iostream>
#include <iomanip>

namespace afwGeom = lsst::afw::geom;

/**
 * Return the transform matrix elements as a parameter vector
 *
 * The elements will be ordered XX, YX, XY, YY
 */
afwGeom::LinearTransform::ParameterVector const afwGeom::LinearTransform::getParameterVector() const {
    ParameterVector r;
    r << (*this)[XX], (*this)[YX], (*this)[XY], (*this)[YY];
    return r;
}

/**
 * Set the transform matrix elements from a parameter vector
 *
 * The parameter vector is ordered XX, YX, XY, YY
 */
void afwGeom::LinearTransform::setParameterVector(
    LinearTransform::ParameterVector const & vector
) {
    (*this)[XX] = vector[XX];  (*this)[XY] = vector[XY];
    (*this)[YX] = vector[YX];  (*this)[YY] = vector[YY];
}

/** 
 * Return the inverse transform. 
 *
 * @throws lsst::afw::geom::SingularTransformException
 */
afwGeom::LinearTransform const afwGeom::LinearTransform::invert() const {
    Eigen::FullPivLU<Matrix> lu(getMatrix());
    if (!lu.isInvertible()) {
        throw LSST_EXCEPT(
            lsst::afw::geom::SingularTransformException,
            "Could not compute LinearTransform inverse"
        );
    }
    Matrix inv = lu.inverse();
    return LinearTransform(inv);
}

/**
 * Return the determinant of the 2x2 matrix
 */
double afwGeom::LinearTransform::computeDeterminant() const {
    Eigen::MatrixXd const & m = getMatrix();
    return m(0, 0)*m(1,1) - m(0,1)*m(1,0);
}

/**
 * Derivative of (*this)(input) with respect to the transform elements (for Point).
 */
afwGeom::LinearTransform::TransformDerivativeMatrix afwGeom::LinearTransform::dTransform(
    Point2D const & input
) const {
    TransformDerivativeMatrix r = TransformDerivativeMatrix::Zero();
    r(0,XX) = input.getX();
    r(0,XY) = input.getY();
    r(1,YX) = input.getX();
    r(1,YY) = input.getY();
    return r;
}

std::ostream& afwGeom::operator<<(
    std::ostream& os, 
    lsst::afw::geom::LinearTransform const & t
) {
    std::ios::fmtflags flags = os.flags();
    int prec = os.precision(7);
    os.setf(std::ios::fixed);
    os << "LinearTransform([(" << std::setw(10) << t[afwGeom::LinearTransform::XX] 
       << "," << std::setw(10) << t[afwGeom::LinearTransform::XY] << "),\n";
    os << "                 (" << std::setw(10) << t[afwGeom::LinearTransform::YX]
       << "," << std::setw(10) << t[afwGeom::LinearTransform::YY] << ")])";
    os.precision(prec);
    os.flags(flags);
    return os;
}


