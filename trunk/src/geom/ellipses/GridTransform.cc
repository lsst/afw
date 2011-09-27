// -*- lsst-c++ -*-

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
#include "lsst/afw/geom/ellipses/GridTransform.h"

namespace lsst { namespace afw { namespace geom {
namespace ellipses {

BaseCore::GridTransform::operator LinearTransform () const {
    double a, b, theta;
    _input._assignToAxes(a, b, theta);
    LinearTransform result = LinearTransform::makeRotation(-theta);
    result[LinearTransform::XX] /= a;
    result[LinearTransform::XY] /= a;
    result[LinearTransform::YX] /= b;
    result[LinearTransform::YY] /= b;
    return result;
}

BaseCore::GridTransform::DerivativeMatrix
BaseCore::GridTransform::d() const {
    double a, b, theta;
    Jacobian rhs = _input._dAssignToAxes(a, b, theta);
    Eigen::Matrix<double,4,3> mid = Eigen::Matrix<double,4,3>::Zero();
    double cos_t = std::cos(theta);
    double sin_t = std::sin(theta);
    mid(LinearTransform::XX, 0) = -cos_t / (a*a);
    mid(LinearTransform::XY, 0) = -sin_t / (a*a);
    mid(LinearTransform::YX, 1) = sin_t / (b*b);
    mid(LinearTransform::YY, 1) = -cos_t / (b*b);
    mid(LinearTransform::XX, 2) = -sin_t / a;
    mid(LinearTransform::XY, 2) = cos_t / a;
    mid(LinearTransform::YX, 2) = -cos_t / b;
    mid(LinearTransform::YY, 2) = -sin_t / b;
    return mid * rhs;

}

double BaseCore::GridTransform::getDeterminant() const {
    double a, b, theta;
    _input._assignToAxes(a, b, theta);
    return 1.0 / (a * b);
}

Ellipse::GridTransform::DerivativeMatrix 
Ellipse::GridTransform::d() const {
    DerivativeMatrix r = DerivativeMatrix::Zero();
    LinearTransform linear = _input.getCore().getGridTransform();
    r.block<4,3>(0,0) = _input.getCore().getGridTransform().d();
    double x = -_input.getCenter().getX();
    double y = -_input.getCenter().getY();
    r(AffineTransform::X, Ellipse::X) = -linear[LinearTransform::XX];
    r(AffineTransform::Y, Ellipse::X) = -linear[LinearTransform::YX];
    r(AffineTransform::X, Ellipse::Y) = -linear[LinearTransform::XY];
    r(AffineTransform::Y, Ellipse::Y) = -linear[LinearTransform::YY];
    r(AffineTransform::X, 0) = x * r(AffineTransform::XX, 0) + y * r(AffineTransform::XY, 0);
    r(AffineTransform::Y, 0) = x * r(AffineTransform::YX, 0) + y * r(AffineTransform::YY, 0);
    r(AffineTransform::X, 1) = x * r(AffineTransform::XX, 1) + y * r(AffineTransform::XY, 1);
    r(AffineTransform::Y, 1) = x * r(AffineTransform::YX, 1) + y * r(AffineTransform::YY, 1);
    r(AffineTransform::X, 2) = x * r(AffineTransform::XX, 2) + y * r(AffineTransform::XY, 2);
    r(AffineTransform::Y, 2) = x * r(AffineTransform::YX, 2) + y * r(AffineTransform::YY, 2);
    return r;
}

Ellipse::GridTransform::operator AffineTransform () const {
    LinearTransform linear = _input.getCore().getGridTransform();
    return AffineTransform(linear, linear(Point2D() - _input.getCenter()));
}

}}}} // namespace lsst::afw::geom::ellipses
