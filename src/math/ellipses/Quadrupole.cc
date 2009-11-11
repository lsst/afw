#include <lsst/afw/math/ellipses/Quadrupole.h>
#include <lsst/afw/math/ellipses/Axes.h>
#include <lsst/afw/math/ellipses/Distortion.h>
#include <lsst/afw/math/ellipses/LogShear.h>

namespace ellipses = lsst::afw::math::ellipses;

ellipses::Quadrupole const & ellipses::QuadrupoleEllipse::getCore() const {
    return static_cast<Quadrupole const&>(*_core);
}

ellipses::Quadrupole & ellipses::QuadrupoleEllipse::getCore() {
    return static_cast<Quadrupole &>(*_core);
}

ellipses::QuadrupoleEllipse::QuadrupoleEllipse(
        lsst::afw::image::PointD const & center = lsst::afw::image::PointD(0,0)
) : Ellipse(new Quadrupole(), center) {}

template <typename Derived>
ellipses::QuadrupoleEllipse::QuadrupoleEllipse(
        Eigen::MatrixBase<Derived> const & vector
) : Ellipse(vector.segment<2>(0)) {
    _core.reset(new Quadrupole(vector.segment<3>(2)));
}
  
ellipses::QuadrupoleEllipse::QuadrupoleEllipse(
    ellipses::Quadrupole const & core, 
    lsst::afw::image::PointD const & center
) : Ellipse(core,center) {}

ellipses::QuadrupoleEllipse::QuadrupoleEllipse(
    ellipses::Ellipse const & other
) : Ellipse(new Quadrupole(other.getCore()), other.getCenter()) {}

ellipses::QuadrupoleEllipse::QuadrupoleEllipse(
    ellipses::QuadrupoleEllipse const & other
) : Ellipse(new Quadrupole(other.getCore()), other.getCenter()) {}





void ellipses::Quadrupole::transform(
    lsst::afw::math::AffineTransform const & transform
) {
    QuadrupoleMatrix matrix = transform.matrix().linear() * getMatrix() 
        * transform.matrix().linear().transpose();
    _vector[IXX] = matrix(0,0);
    _vector[IYY] = matrix(1,1);
    _vector[IXY] = matrix(0,1);
}

ellipses::Quadrupole::Ellipse * ellipses::Quadrupole::makeEllipse(
    lsst::afw::image::PointD const & center
) const {
    return new Ellipse(*this,center);
}

void ellipses::Quadrupole::assignTo(ellipses::Quadrupole & other) const {
    other._vector = this->_vector;
}

void ellipses::Quadrupole::assignTo(ellipses::Axes & other) const {
    double xx_p_yy = _vector[IXX] + _vector[IYY];
    double xx_m_yy = _vector[IXX] - _vector[IYY];
    double t = std::sqrt(xx_m_yy*xx_m_yy + 4*_vector[IXY]*_vector[IXY]);
    other[Axes::A] = std::sqrt(0.5*(xx_p_yy+t));
    other[Axes::B] = std::sqrt(0.5*(xx_p_yy-t));
    other[Axes::THETA] = 0.5*std::atan2(2.0*_vector[IXY],xx_m_yy);
}

void ellipses::Quadrupole::assignTo(ellipses::Distortion & other) const {
    double t = _vector[IXX] + _vector[IYY];
    if (t < 1E-12) {
    	other[Distortion::E1] = 0.0;
        other[Distortion::E2] = 0.0;
        other[Distortion::R] = 0.0;
    } else {
	    other[Distortion::E1] = (_vector[IXX]-_vector[IYY])/t;
	    other[Distortion::E2] = 2*_vector[IXY]/t;
        other[Distortion::R] = std::pow(getDeterminant(),0.25);
    }
}

void ellipses::Quadrupole::assignTo(ellipses::LogShear & other) const {
    Distortion tmp;
    this->assignTo(tmp);
    static_cast<Core&>(tmp).assignTo(other);
}

Eigen::Matrix3d ellipses::Quadrupole::differentialAssignTo(
    ellipses::Quadrupole & other
) const {
    other._vector = this->_vector;
    return Eigen::Matrix3d::Identity();
}

Eigen::Matrix3d ellipses::Quadrupole::differentialAssignTo(
    ellipses::Axes & other
) const {
    Distortion tmp;
    Eigen::Matrix3d a = this->differentialAssignTo(tmp);
    Eigen::Matrix3d b = static_cast<Core&>(tmp).differentialAssignTo(other);
    return b*a;
}

Eigen::Matrix3d ellipses::Quadrupole::differentialAssignTo(
    ellipses::Distortion & other
) const {
    assignTo(other);
    Eigen::Matrix3d m;
    double d = 1.0 / (_vector[IXX] + _vector[IYY]);
    m(0,2) = 0.0;
    m(1,2) = 2.0 * d;
    m.corner<2,2>(Eigen::TopLeft).setConstant(2.0*d*d);
    m.row(2).setConstant(std::pow(getDeterminant(),-0.75));
    m(0,0) *= _vector[IYY];
    m(0,1) *= -_vector[IXX];
    m(1,0) *= -_vector[IXY];
    m(1,1) *= -_vector[IXY];
    m(2,0) *= _vector[IYY]/4;
    m(2,1) *= _vector[IXX]/4;
    m(2,2) *= -_vector[IXY]/2;
    return m;
}

Eigen::Matrix3d ellipses::Quadrupole::differentialAssignTo(
    ellipses::LogShear & other
) const {
    Distortion tmp;
    Eigen::Matrix3d a = this->differentialAssignTo(tmp);
    Eigen::Matrix3d b = static_cast<Core&>(tmp).differentialAssignTo(other);
    return b*a;
}
