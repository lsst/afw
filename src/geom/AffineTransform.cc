/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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

#include "Eigen/LU"

#include <iomanip>

#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/pex/exceptions/Runtime.h"

namespace lsst {
namespace afw {
namespace geom {

AffineTransform::ParameterVector const AffineTransform::getParameterVector() const {
    ParameterVector r;
    r << (*this)[XX], (*this)[YX], (*this)[XY], (*this)[YY], (*this)[X], (*this)[Y];
    return r;
}

void AffineTransform::setParameterVector(AffineTransform::ParameterVector const &vector) {
    (*this)[XX] = vector[XX];
    (*this)[XY] = vector[XY];
    (*this)[X] = vector[X];
    (*this)[YX] = vector[YX];
    (*this)[YY] = vector[YY];
    (*this)[Y] = vector[Y];
}

AffineTransform::Matrix const AffineTransform::getMatrix() const {
    Matrix r;
    r << (*this)[XX], (*this)[XY], (*this)[X], (*this)[YX], (*this)[YY], (*this)[Y], 0.0, 0.0, 1.0;
    return r;
}

AffineTransform const AffineTransform::invert() const {
    LinearTransform inv(getLinear().invert());
    return AffineTransform(inv, -inv(getTranslation()));
}

AffineTransform::TransformDerivativeMatrix AffineTransform::dTransform(Point2D const &input) const {
    TransformDerivativeMatrix r = TransformDerivativeMatrix::Zero();
    r.block<2, 4>(0, 0) = getLinear().dTransform(input);
    r(0, X) = 1.0;
    r(1, Y) = 1.0;
    return r;
}

AffineTransform::TransformDerivativeMatrix AffineTransform::dTransform(Extent2D const &input) const {
    TransformDerivativeMatrix r = TransformDerivativeMatrix::Zero();
    r.block<2, 4>(0, 0) = getLinear().dTransform(input);
    return r;
}

std::ostream &operator<<(std::ostream &os, AffineTransform const &transform) {
    std::ios::fmtflags flags = os.flags();
    AffineTransform::Matrix const &matrix = transform.getMatrix();
    int prec = os.precision(7);
    os.setf(std::ios::fixed);
    os << "AffineTransform([(" << std::setw(10) << matrix(0, 0) << "," << std::setw(10) << matrix(0, 1) << ","
       << std::setw(10) << matrix(0, 2) << "),\n";
    os << "                 (" << std::setw(10) << matrix(1, 0) << "," << std::setw(10) << matrix(1, 1) << ","
       << std::setw(10) << matrix(1, 2) << "),\n";
    os << "                 (" << std::setw(10) << matrix(2, 0) << "," << std::setw(10) << matrix(2, 1) << ","
       << std::setw(10) << matrix(2, 2) << ")])";
    os.precision(prec);
    os.flags(flags);
    return os;
}

AffineTransform makeAffineTransformFromTriple(Point2D const &p1, Point2D const &p2, Point2D const &p3,
                                              Point2D const &q1, Point2D const &q2, Point2D const &q3) {
    Eigen::Matrix3d mp;
    mp << p1.getX(), p2.getX(), p3.getX(), p1.getY(), p2.getY(), p3.getY(), 1.0, 1.0, 1.0;

    Eigen::Matrix3d mq;
    mq << q1.getX(), q2.getX(), q3.getX(), q1.getY(), q2.getY(), q3.getY(), 1.0, 1.0, 1.0;

    Eigen::Matrix3d m = mq * mp.inverse();
    return AffineTransform(m);
}
}
}
}  // end lsst::afw::geom
