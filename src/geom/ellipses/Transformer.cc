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
#include "lsst/afw/geom/ellipses/Transformer.h"

#include "Eigen/LU"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

BaseCore::Ptr BaseCore::Transformer::copy() const {
    BaseCore::Ptr r(input.clone());
    apply(*r);
    return r;
}

void BaseCore::Transformer::inPlace() {
    apply(input);
}

void BaseCore::Transformer::apply(BaseCore & result) const {
    Eigen::Matrix2d m;
    input._assignToQuadrupole(m(0,0), m(1,1), m(0,1));
    m(1,0) = m(0,1);
    m = transform.getMatrix() * m * transform.getMatrix().transpose();
    result._assignFromQuadrupole(m(0,0), m(1,1), m(0,1));
}

BaseCore::Transformer::DerivativeMatrix
BaseCore::Transformer::d() const {
    BaseCore::Ptr output(input.clone());
    Eigen::Matrix2d m;
    Jacobian rhs = input._dAssignToQuadrupole(m(0,0), m(1,1), m(0,1));
    m(1,0) = m(0,1);
    m = transform.getMatrix() * m * transform.getMatrix().transpose();
    Jacobian lhs = output->_dAssignFromQuadrupole(m(0,0), m(1,1), m(0,1));
    Jacobian mid = Jacobian::Zero();
    mid(0,0) = transform[LinearTransform::XX]*transform[LinearTransform::XX];
    mid(0,1) = transform[LinearTransform::XY]*transform[LinearTransform::XY];
    mid(0,2) = 2*transform[LinearTransform::XY]*transform[LinearTransform::XX];
    mid(1,0) = transform[LinearTransform::YX]*transform[LinearTransform::YX];
    mid(1,1) = transform[LinearTransform::YY]*transform[LinearTransform::YY];
    mid(1,2) = 2*transform[LinearTransform::YY]*transform[LinearTransform::YX];
    mid(2,0) = transform[LinearTransform::YX]*transform[LinearTransform::XX];
    mid(2,1) = transform[LinearTransform::YY]*transform[LinearTransform::XY];
    mid(2,2) = transform[LinearTransform::XX]*transform[LinearTransform::YY]
        + transform[LinearTransform::XY]*transform[LinearTransform::YX];
    return lhs * mid * rhs;
}

BaseCore::Transformer::TransformDerivativeMatrix
BaseCore::Transformer::dTransform() const {
    BaseCore::Ptr output(input.clone());
    Eigen::Matrix2d m;
    input._assignToQuadrupole(m(0,0), m(1,1), m(0,1));
    Eigen::Matrix<double,3,4> mid = Eigen::Matrix<double,3,4>::Zero();
    m(1,0) = m(0,1);
    mid(0, LinearTransform::XX) =
        2.0*(transform[LinearTransform::XX]*m(0,0) + transform[LinearTransform::XY]*m(0,1));
    mid(0, LinearTransform::XY) =
        2.0*(transform[LinearTransform::XX]*m(0,1) + transform[LinearTransform::XY]*m(1,1));
    mid(1, LinearTransform::YX) =
        2.0*(transform[LinearTransform::YX]*m(0,0) + transform[LinearTransform::YY]*m(0,1));
    mid(1, LinearTransform::YY) =
        2.0*(transform[LinearTransform::YX]*m(0,1) + transform[LinearTransform::YY]*m(1,1));
    mid(2, LinearTransform::XX) =
        transform[LinearTransform::YX]*m(0,0) + transform[LinearTransform::YY]*m(0,1);
    mid(2, LinearTransform::XY) =
        transform[LinearTransform::YX]*m(0,1) + transform[LinearTransform::YY]*m(1,1);
    mid(2, LinearTransform::YX) =
        transform[LinearTransform::XX]*m(0,0) + transform[LinearTransform::XY]*m(0,1);
    mid(2, LinearTransform::YY) =
        transform[LinearTransform::XX]*m(0,1) + transform[LinearTransform::XY]*m(1,1);
    m = transform.getMatrix() * m * transform.getMatrix().transpose();
    Jacobian lhs = output->_dAssignFromQuadrupole(m(0,0), m(1,1), m(0,1));
    return lhs * mid;
}

Ellipse::Ptr Ellipse::Transformer::copy() const {
    Ellipse::Ptr r = std::make_shared<Ellipse>(
        input.getCore().transform(transform.getLinear()).copy(),
        transform(input.getCenter())
    );
    return r;
}

void Ellipse::Transformer::inPlace() {
    input.setCenter(transform(input.getCenter()));
    input.getCore().transform(transform.getLinear()).inPlace();
}

Ellipse::Transformer::DerivativeMatrix
Ellipse::Transformer::d() const {
    DerivativeMatrix r = DerivativeMatrix::Zero();
    r.block<2,2>(3,3) = transform.getLinear().getMatrix();
    r.block<3,3>(0,0) = input.getCore().transform(transform.getLinear()).d();
    return r;
}

Ellipse::Transformer::TransformDerivativeMatrix
Ellipse::Transformer::dTransform() const {
    TransformDerivativeMatrix r = TransformDerivativeMatrix::Zero();
    r.block<2,6>(3,0) = transform.dTransform(input.getCenter());
    r.block<3,4>(0,0) = input.getCore().transform(transform.getLinear()).dTransform();
    return r;
}

}}}} // namespace lsst::afw::geom::ellipses
