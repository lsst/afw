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
 
#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/afw/geom/ellipses/Transformer.h"

#include <Eigen/LU>

namespace ellipses = lsst::afw::geom::ellipses;

ellipses::Quadrupole
ellipses::BaseCore::Transformer::transformQuadrupole(Quadrupole const & quadrupole) const {
    Quadrupole::Matrix matrix = _transform.getMatrix() * 
        quadrupole.getMatrix() * 
        _transform.getMatrix().transpose();
    return Quadrupole(matrix(0,0), matrix(1,1), matrix(0,1));
}

boost::tuple<ellipses::Quadrupole,ellipses::Quadrupole,
             ellipses::BaseCore::Jacobian,ellipses::BaseCore::Jacobian>
ellipses::BaseCore::Transformer::computeConversionJacobian() const {
    Quadrupole inputQuadrupole;
    BaseCore::Jacobian jacobian = inputQuadrupole.dAssign(_input);
    Quadrupole outputQuadrupole = transformQuadrupole(inputQuadrupole);
    boost::shared_ptr<BaseCore> tmp(_input.clone());
    BaseCore::Jacobian jacobian_inv = tmp->dAssign(outputQuadrupole);
    return boost::make_tuple(inputQuadrupole,outputQuadrupole,jacobian,jacobian_inv);
}

boost::shared_ptr<ellipses::BaseCore> ellipses::BaseCore::Transformer::copy() const {
    boost::shared_ptr<BaseCore> r(_input.clone());
    *r = transformQuadrupole(_input);
    return r;
}

ellipses::BaseCore::Transformer::DerivativeMatrix
ellipses::BaseCore::Transformer::d() const {
    DerivativeMatrix r;
    r(Quadrupole::IXX,Quadrupole::IXX) = _transform[LinearTransform::XX]*_transform[LinearTransform::XX];
    r(Quadrupole::IXX,Quadrupole::IYY) = _transform[LinearTransform::XY]*_transform[LinearTransform::XY];
    r(Quadrupole::IXX,Quadrupole::IXY) = 2*_transform[LinearTransform::XY]*_transform[LinearTransform::XX];
    r(Quadrupole::IYY,Quadrupole::IXX) = _transform[LinearTransform::YX]*_transform[LinearTransform::YX];
    r(Quadrupole::IYY,Quadrupole::IYY) = _transform[LinearTransform::YY]*_transform[LinearTransform::YY];
    r(Quadrupole::IYY,Quadrupole::IXY) = 2*_transform[LinearTransform::YY]*_transform[LinearTransform::YX];
    r(Quadrupole::IXY,Quadrupole::IXX) = _transform[LinearTransform::YX]*_transform[LinearTransform::XX];
    r(Quadrupole::IXY,Quadrupole::IYY) = _transform[LinearTransform::YY]*_transform[LinearTransform::XY];
    r(Quadrupole::IXY,Quadrupole::IXY) = _transform[LinearTransform::XX]*_transform[LinearTransform::YY] 
        + _transform[LinearTransform::XY]*_transform[LinearTransform::YX];
    BaseCore::Jacobian j;
    BaseCore::Jacobian j_inv;
    boost::tie(boost::tuples::ignore,boost::tuples::ignore,j,j_inv) = computeConversionJacobian();
    return j_inv * r * j;
}

ellipses::BaseCore::Transformer::TransformDerivativeMatrix
ellipses::BaseCore::Transformer::dTransform() const {
    TransformDerivativeMatrix r = TransformDerivativeMatrix::Zero();
    Quadrupole inputQuadrupole;
    BaseCore::Jacobian j_inv;
    boost::tie(inputQuadrupole,boost::tuples::ignore,boost::tuples::ignore,j_inv) 
        = computeConversionJacobian();
    double xx = inputQuadrupole[Quadrupole::IXX];
    double yy = inputQuadrupole[Quadrupole::IYY];
    double xy = inputQuadrupole[Quadrupole::IXY];
    r(Quadrupole::IXX,LinearTransform::XX) 
        = 2*(_transform[LinearTransform::XX]*xx + _transform[LinearTransform::XY]*xy);
    r(Quadrupole::IXX,LinearTransform::XY)
        = 2*(_transform[LinearTransform::XX]*xy + _transform[LinearTransform::XY]*yy);
    r(Quadrupole::IYY,LinearTransform::YX)
        = 2*(_transform[LinearTransform::YX]*xx + _transform[LinearTransform::YY]*xy);
    r(Quadrupole::IYY,LinearTransform::YY)
        = 2*(_transform[LinearTransform::YX]*xy + _transform[LinearTransform::YY]*yy);
    r(Quadrupole::IXY,LinearTransform::XX)
        = _transform[LinearTransform::YX]*xx + _transform[LinearTransform::YY]*xy;
    r(Quadrupole::IXY,LinearTransform::XY)
        = _transform[LinearTransform::YX]*xy + _transform[LinearTransform::YY]*yy;
    r(Quadrupole::IXY,LinearTransform::YX)
        = _transform[LinearTransform::XX]*xx + _transform[LinearTransform::XY]*xy;
    r(Quadrupole::IXY,LinearTransform::YY)
        = _transform[LinearTransform::XX]*xy + _transform[LinearTransform::XY]*yy;
    return  j_inv * r;
}

boost::shared_ptr<ellipses::BaseEllipse> ellipses::BaseEllipse::Transformer::copy() const {
    boost::shared_ptr<BaseEllipse> r(_input.clone());
    r->setCenter(_transform(r->getCenter()));
    r->getCore().transform(_transform.getLinear()).inPlace();
    return r;
}

void ellipses::BaseEllipse::Transformer::inPlace() {
    _input.setCenter(_transform(_input.getCenter()));
    _input.getCore().transform(_transform.getLinear()).inPlace();
}

ellipses::BaseEllipse::Transformer::DerivativeMatrix 
ellipses::BaseEllipse::Transformer::d() const {
    DerivativeMatrix r = DerivativeMatrix::Zero();
    r.block<2,2>(0,0) = _transform.getLinear().getMatrix();
    r.block<3,3>(2,2) = _input.getCore().transform(_transform.getLinear()).d();
    return r;
}

ellipses::BaseEllipse::Transformer::TransformDerivativeMatrix
ellipses::BaseEllipse::Transformer::dTransform() const {
    TransformDerivativeMatrix r = TransformDerivativeMatrix::Zero();
    r.block<2,6>(0,0) = _transform.dTransform(_input.getCenter());
    r.block<3,4>(2,0) = _input.getCore().transform(_transform.getLinear()).dTransform();
    return r;
}
