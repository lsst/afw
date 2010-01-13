// -*- lsst-c++ -*-
#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/afw/geom/ellipses/Transformer.h"

#include <Eigen/LU>

namespace ellipses = lsst::afw::geom::ellipses;

ellipses::Quadrupole
ellipses::BaseCore::Transformer::transformQuadrupole(Quadrupole const & quadrupole) const {
    Quadrupole::Matrix matrix = _transform.getEigenTransform().linear() * quadrupole.getMatrix() 
        * _transform.getEigenTransform().linear().transpose();
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
    DerivativeMatrix r = DerivativeMatrix::Zero();
    r(Quadrupole::IXX,Quadrupole::IXX) = _transform[AffineTransform::XX]*_transform[AffineTransform::XX];
    r(Quadrupole::IXX,Quadrupole::IYY) = _transform[AffineTransform::XY]*_transform[AffineTransform::XY];
    r(Quadrupole::IXX,Quadrupole::IXY) = 2*_transform[AffineTransform::XY]*_transform[AffineTransform::XX];
    r(Quadrupole::IYY,Quadrupole::IXX) = _transform[AffineTransform::YX]*_transform[AffineTransform::YX];
    r(Quadrupole::IYY,Quadrupole::IYY) = _transform[AffineTransform::YY]*_transform[AffineTransform::YY];
    r(Quadrupole::IYY,Quadrupole::IXY) = 2*_transform[AffineTransform::YY]*_transform[AffineTransform::YX];
    r(Quadrupole::IXY,Quadrupole::IXX) = _transform[AffineTransform::YX]*_transform[AffineTransform::XX];
    r(Quadrupole::IXY,Quadrupole::IYY) = _transform[AffineTransform::YY]*_transform[AffineTransform::XY];
    r(Quadrupole::IXY,Quadrupole::IXY) = _transform[AffineTransform::XX]*_transform[AffineTransform::YY] 
        + _transform[AffineTransform::XY]*_transform[AffineTransform::YX];
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
    r(Quadrupole::IXX,AffineTransform::XX) 
        = 2*(_transform[AffineTransform::XX]*xx + _transform[AffineTransform::XY]*xy);
    r(Quadrupole::IXX,AffineTransform::XY)
        = 2*(_transform[AffineTransform::XX]*xy + _transform[AffineTransform::XY]*yy);
    r(Quadrupole::IYY,AffineTransform::YX)
        = 2*(_transform[AffineTransform::YX]*xx + _transform[AffineTransform::YY]*xy);
    r(Quadrupole::IYY,AffineTransform::YY)
        = 2*(_transform[AffineTransform::YX]*xy + _transform[AffineTransform::YY]*yy);
    r(Quadrupole::IXY,AffineTransform::XX)
        = _transform[AffineTransform::YX]*xx + _transform[AffineTransform::YY]*xy;
    r(Quadrupole::IXY,AffineTransform::XY)
        = _transform[AffineTransform::YX]*xy + _transform[AffineTransform::YY]*yy;
    r(Quadrupole::IXY,AffineTransform::YX)
        = _transform[AffineTransform::XX]*xx + _transform[AffineTransform::XY]*xy;
    r(Quadrupole::IXY,AffineTransform::YY)
        = _transform[AffineTransform::XX]*xy + _transform[AffineTransform::XY]*yy;
    return  j_inv * r;
}

boost::shared_ptr<ellipses::BaseEllipse> ellipses::BaseEllipse::Transformer::copy() const {
    boost::shared_ptr<BaseEllipse> r(_input.clone());
    r->setCenter(_transform(r->getCenter()));
    r->getCore().transform(_transform).inPlace();
    return r;
}

void ellipses::BaseEllipse::Transformer::inPlace() {
    _input.setCenter(_transform(_input.getCenter()));
    _input.getCore().transform(_transform).inPlace();
}

ellipses::BaseEllipse::Transformer::DerivativeMatrix 
ellipses::BaseEllipse::Transformer::d() const {
    DerivativeMatrix r = DerivativeMatrix::Zero();
    r.block<2,2>(0,0) = _transform.getEigenTransform().linear();
    r.block<3,3>(2,2) = _input.getCore().transform(_transform).d();
    return r;
}

ellipses::BaseEllipse::Transformer::TransformDerivativeMatrix
ellipses::BaseEllipse::Transformer::dTransform() const {
    TransformDerivativeMatrix r = TransformDerivativeMatrix::Zero();
    r.block<2,6>(0,0) = _transform.dTransform(_input.getCenter());
    r.block<3,6>(2,0) = _input.getCore().transform(_transform).dTransform();
    return r;
}
