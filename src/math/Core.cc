#include <lsst/afw/math/ellipses/Quadrupole.h>
#include <lsst/afw/math/ellipses/Axes.h>

#include <Eigen/LU>

namespace ellipses = lsst::afw::math::ellipses;

ellipses::Core::TransformDerivative::TransformDerivative(
    ellipses::Core const & core, 
    lsst::afw::math::AffineTransform const & transform
) : _transform(transform) {
    Quadrupole tmp;
    _jacobian = tmp.differentialAssign(core);
    _quadrupole = tmp.getVector();
    tmp.transform(transform);
    std::auto_ptr<Core> new_core(core.clone());
    _jacobian_inv = new_core->differentialAssign(tmp);
}

Eigen::Matrix3d ellipses::Core::TransformDerivative::dInput() const {
    Eigen::Matrix3d r = Eigen::Matrix3d::Zero();
    r(Quadrupole::IXX,Quadrupole::IXX) = 
            _transform[AffineTransform::XX] * _transform[AffineTransform::XX];
    r(Quadrupole::IXX,Quadrupole::IYY) = 
            _transform[AffineTransform::XY] * _transform[AffineTransform::XY];
    r(Quadrupole::IXX,Quadrupole::IXY) = 2 * 
            _transform[AffineTransform::XY] * _transform[AffineTransform::XX];
    r(Quadrupole::IYY,Quadrupole::IXX) = 
            _transform[AffineTransform::YX] * _transform[AffineTransform::YX];
    r(Quadrupole::IYY,Quadrupole::IYY) = 
            _transform[AffineTransform::YY] * _transform[AffineTransform::YY];
    r(Quadrupole::IYY,Quadrupole::IXY) = 2 * 
            _transform[AffineTransform::YY] * _transform[AffineTransform::YX];
    r(Quadrupole::IXY,Quadrupole::IXX) = 
            _transform[AffineTransform::YX] * _transform[AffineTransform::XX];
    r(Quadrupole::IXY,Quadrupole::IYY) = 
            _transform[AffineTransform::YY] * _transform[AffineTransform::XY];
    r(Quadrupole::IXY,Quadrupole::IXY) = 
            _transform[AffineTransform::XX] * _transform[AffineTransform::YY] 
          + _transform[AffineTransform::XY] * _transform[AffineTransform::YX];

    return _jacobian_inv * r * _jacobian;
}

Eigen::Matrix<double,3,6> ellipses::Core::TransformDerivative::dTransform() const {
    Eigen::Matrix<double,3,6> r = Eigen::Matrix<double,3,6>::Zero();
    double xx = _quadrupole[Quadrupole::IXX];
    double yy = _quadrupole[Quadrupole::IYY];
    double xy = _quadrupole[Quadrupole::IXY];
    r(Quadrupole::IXX,AffineTransform::XX) = 2 * (
            _transform[AffineTransform::XX] * xx + 
            _transform[AffineTransform::XY] * xy
    );
    r(Quadrupole::IXX,AffineTransform::XY) = 2 * (
            _transform[AffineTransform::XX] * xy + 
            _transform[AffineTransform::XY] * yy
    );
    r(Quadrupole::IYY,AffineTransform::YX) = 2 * (
            _transform[AffineTransform::YX] * xx + 
            _transform[AffineTransform::YY] * xy
    );
    r(Quadrupole::IYY,AffineTransform::YY) = 2 * (
            _transform[AffineTransform::YX] * xy + 
            _transform[AffineTransform::YY] * yy
    );
    r(Quadrupole::IXY,AffineTransform::XX) = 
            _transform[AffineTransform::YX] * xx + 
            _transform[AffineTransform::YY] * xy;
    r(Quadrupole::IXY,AffineTransform::XY) = 
            _transform[AffineTransform::YX] * xy + 
            _transform[AffineTransform::YY] * yy;
    r(Quadrupole::IXY,AffineTransform::YX) =             
            _transform[AffineTransform::XX] * xx + 
            _transform[AffineTransform::XY] * xy;
    r(Quadrupole::IXY,AffineTransform::YY) = 
            _transform[AffineTransform::XX] * xy + 
            _transform[AffineTransform::XY] * yy;

    return  _jacobian_inv * r;
}

void ellipses::Core::transform(
    lsst::afw::math::AffineTransform const & transform
) {
    Quadrupole tmp(*this);
    tmp.transform(transform);
    *this = tmp;
}

lsst::afw::math::AffineTransform ellipses::Core::getGenerator() const {
    Axes tmp(*this);
    return tmp.getGenerator();
}
