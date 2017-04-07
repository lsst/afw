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

#include "lsst/afw/geom/LinearTransform.h"

#include "Eigen/LU"

#include <iostream>
#include <iomanip>

namespace afwGeom = lsst::afw::geom;

afwGeom::LinearTransform::ParameterVector const afwGeom::LinearTransform::getParameterVector() const {
    ParameterVector r;
    r << (*this)[XX], (*this)[YX], (*this)[XY], (*this)[YY];
    return r;
}

void afwGeom::LinearTransform::setParameterVector(
    LinearTransform::ParameterVector const & vector
) {
    (*this)[XX] = vector[XX];  (*this)[XY] = vector[XY];
    (*this)[YX] = vector[YX];  (*this)[YY] = vector[YY];
}

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

double afwGeom::LinearTransform::computeDeterminant() const {
    Eigen::MatrixXd const & m = getMatrix();
    return m(0, 0)*m(1,1) - m(0,1)*m(1,0);
}

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


