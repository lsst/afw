// -*- lsst-c++ -*-
#include "lsst/afw/geom/ellipses/RadialFraction.h"
#include <Eigen/LU>

namespace ellipses = lsst::afw::geom::ellipses;

ellipses::BaseCore::RadialFraction::RadialFraction(BaseCore const & core) 
    : _inv_matrix(), _jacobian(Eigen::Matrix3d::Identity()) {
    Quadrupole tmp;
    _jacobian = tmp.dAssign(core);
    _inv_matrix = tmp.getMatrix().inverse();
}

ellipses::BaseCore::RadialFraction::DerivativeVector
ellipses::BaseCore::RadialFraction::d(PointD const & p) const {
    DerivativeVector v(_inv_matrix*p.asVector());
    return v /= sqrt(p.asVector().dot(v));
}

ellipses::BaseCore::RadialFraction::CoreDerivativeVector
ellipses::BaseCore::RadialFraction::dCore(PointD const & p) const {
    CoreDerivativeVector vec;
    Eigen::Vector2d tmp1 = _inv_matrix * p.asVector();
    Quadrupole::Matrix tmp2;
    tmp2.part<Eigen::SelfAdjoint>() = (tmp1 * tmp1.adjoint()).lazy();
    vec[0] = -0.5*tmp2(0,0);
    vec[1] = -0.5*tmp2(1,1);
    vec[2] = -tmp2(1,0);
    vec /= std::sqrt(p.asVector().dot(tmp1));
    return vec * _jacobian;
}

ellipses::BaseEllipse::RadialFraction::EllipseDerivativeVector 
ellipses::BaseEllipse::RadialFraction::dEllipse(PointD const & p) const {
    EllipseDerivativeVector vec;
    vec.segment<2>(0) = - d(p);
    vec.segment<3>(2) = _coreRF.dCore(p - _offset);
    return vec;
}
