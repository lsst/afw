// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
#include "lsst/afw/geom/ellipses/radii.h"
#include <complex>

namespace lsst { namespace afw { namespace geom {
namespace ellipses {

void DeterminantRadius::assignFromQuadrupole(
    double ixx, double iyy, double ixy, 
    Distortion & distortion
) {
    _value = std::pow(ixx * iyy - ixy * ixy, 0.25);
    distortion.setE1((ixx - iyy) / (ixx + iyy));
    distortion.setE2(2.0 * ixy / (ixx + iyy));
}

BaseCore::Jacobian DeterminantRadius::dAssignFromQuadrupole(
    double ixx, double iyy, double ixy, 
    Distortion & distortion
) {
    double xx_yy = ixx + iyy;
    BaseCore::Jacobian result = BaseCore::Jacobian::Zero();
    _value = std::pow(ixx * iyy - ixy * ixy, 0.25);
    distortion.setE1((ixx - iyy) / xx_yy);
    distortion.setE2(2.0 * ixy / xx_yy);
    result.block<2,2>(0,0).setConstant(2.0 / (xx_yy * xx_yy));
    result(0, 0) *= iyy;
    result(0, 1) *= -ixx;
    result(1, 0) *= -ixy;
    result(1, 1) *= -ixy;
    result(1, 2) = 2.0 / xx_yy;
    result.row(2).setConstant(1.0 / (4.0 * _value * _value * _value));
    result(2, 0) *= iyy;
    result(2, 1) *= ixx;
    result(2, 2) *= -2.0*ixy;
    return result;
}

void DeterminantRadius::assignToQuadrupole(
    Distortion const & distortion,
    double & ixx, double & iyy, double & ixy
) const {
    double den = std::sqrt(1.0 - std::norm(distortion.getComplex()));
    double r2 = _value * _value;
    ixx = r2 * (1.0 + distortion.getE1()) / den;
    iyy = r2 * (1.0 - distortion.getE1()) / den;
    ixy = r2 * distortion.getE2() / den;
}


BaseCore::Jacobian DeterminantRadius::dAssignToQuadrupole(
    Distortion const & distortion,
    double & ixx, double & iyy, double & ixy
) const {
    BaseCore::Jacobian result = BaseCore::Jacobian::Zero();
    double den = std::sqrt(1.0 - std::norm(distortion.getComplex()));
    result.col(2).setConstant(2.0 * _value / den);
    double r2 = _value * _value;
    ixx = r2 * (1.0 + distortion.getE1()) / den;
    iyy = r2 * (1.0 - distortion.getE1()) / den;
    ixy = r2 * distortion.getE2() / den;
    result.block<3,2>(0, 0).setConstant(r2 / (den * den * den));
    result(0, 0) *= (distortion.getE1() + 1.0 - distortion.getE2() * distortion.getE2());
    result(1, 0) *= (distortion.getE1() - 1.0 + distortion.getE2() * distortion.getE2());
    result(2, 0) *= distortion.getE1() * distortion.getE2();
    result(0, 1) *= distortion.getE2() * (1.0 + distortion.getE1());
    result(1, 1) *= distortion.getE2() * (1.0 - distortion.getE1());
    result(2, 1) *= (1.0 - distortion.getE1() * distortion.getE1());
    result(0, 2) *= (1.0 + distortion.getE1());
    result(1, 2) *= (1.0 - distortion.getE1());
    result(2, 2) *= distortion.getE2();
    return result;
}

void TraceRadius::assignFromQuadrupole(
    double ixx, double iyy, double ixy, 
    Distortion & distortion
) {
    _value = std::sqrt(0.5 * (ixx + iyy));
    distortion.setE1((ixx - iyy) / (ixx + iyy));
    distortion.setE2(2.0 * ixy / (ixx + iyy));
}

BaseCore::Jacobian TraceRadius::dAssignFromQuadrupole(
    double ixx, double iyy, double ixy, 
    Distortion & distortion
) {
    double xx_yy = ixx + iyy;
    BaseCore::Jacobian result = BaseCore::Jacobian::Zero();
    _value = std::sqrt(0.5 * xx_yy);
    distortion.setE1((ixx - iyy) / xx_yy);
    distortion.setE2(2.0 * ixy / xx_yy);
    result.block<2,2>(0,0).setConstant(2.0 / (xx_yy * xx_yy));
    result(0, 0) *= iyy;
    result(0, 1) *= -ixx;
    result(1, 0) *= -ixy;
    result(1, 1) *= -ixy;
    result(1, 2) = 2.0 / xx_yy;
    result(2, 0) = 0.25 / _value;
    result(2, 1) = 0.25 / _value;
    return result;
}

void TraceRadius::assignToQuadrupole(
    Distortion const & distortion,
    double & ixx, double & iyy, double & ixy
) const {
    double r2 = _value * _value;
    ixx = r2 * (1.0 + distortion.getE1());
    iyy = r2 * (1.0 - distortion.getE1());
    ixy = r2 * distortion.getE2();
}

BaseCore::Jacobian TraceRadius::dAssignToQuadrupole(
    Distortion const & distortion,
    double & ixx, double & iyy, double & ixy
) const {
    BaseCore::Jacobian result = BaseCore::Jacobian::Zero();
    result.col(2).setConstant(2.0 * _value);
    result(0, 2) *= (1.0 + distortion.getE1());
    result(1, 2) *= (1.0 - distortion.getE1());
    result(2, 2) *= distortion.getE2();
    double r2 = _value * _value;
    ixx = r2 * (1.0 + distortion.getE1());
    iyy = r2 * (1.0 - distortion.getE1());
    ixy = r2 * distortion.getE2(); 
    result(0, 0) = r2;
    result(1, 0) = -r2;
    result(2, 1) = r2;
    return result;
}

void LogDeterminantRadius::assignFromQuadrupole(
    double ixx, double iyy, double ixy, 
    Distortion & distortion
) {
    _value = 0.25 * std::log(ixx * iyy - ixy * ixy);
    distortion.setE1((ixx - iyy) / (ixx + iyy));
    distortion.setE2(2.0 * ixy / (ixx + iyy));
}

BaseCore::Jacobian LogDeterminantRadius::dAssignFromQuadrupole(
    double ixx, double iyy, double ixy, 
    Distortion & distortion
) {
    double xx_yy = ixx + iyy;
    BaseCore::Jacobian result = BaseCore::Jacobian::Zero();
    double det = ixx * iyy - ixy * ixy;
    _value = 0.25 * std::log(det);
    distortion.setE1((ixx - iyy) / xx_yy);
    distortion.setE2(2.0 * ixy / xx_yy);
    result.block<2,2>(0,0).setConstant(2.0 / (xx_yy * xx_yy));
    result(0, 0) *= iyy;
    result(0, 1) *= -ixx;
    result(1, 0) *= -ixy;
    result(1, 1) *= -ixy;
    result(1, 2) = 2.0 / xx_yy;
    result(2, 0) = 0.25 * iyy / det;
    result(2, 1) = 0.25 * ixx / det;
    result(2, 2) = -0.5 * ixy / det;
    return result;
}

void LogDeterminantRadius::assignToQuadrupole(
    Distortion const & distortion,
    double & ixx, double & iyy, double & ixy
) const {
    double den = std::sqrt(1.0 - std::norm(distortion.getComplex()));
    double r2 = std::exp(2.0 * _value);
    ixx = r2 * (1.0 + distortion.getE1()) / den;
    iyy = r2 * (1.0 - distortion.getE1()) / den;
    ixy = r2 * distortion.getE2() / den;
}

BaseCore::Jacobian LogDeterminantRadius::dAssignToQuadrupole(
    Distortion const & distortion,
    double & ixx, double & iyy, double & ixy
) const {
    BaseCore::Jacobian result = BaseCore::Jacobian::Zero();
    double den = std::sqrt(1.0 - std::norm(distortion.getComplex()));
    result.col(2).setConstant(2.0 * _value / den);
    double r2 = std::exp(2.0 * _value);
    ixx = r2 * (1.0 + distortion.getE1()) / den;
    iyy = r2 * (1.0 - distortion.getE1()) / den;
    ixy = r2 * distortion.getE2() / den;
    result.block<3,2>(0, 0).setConstant(r2 / (den * den * den));
    result(0, 0) *= distortion.getE1() + 1.0 - distortion.getE2() * distortion.getE2();
    result(1, 0) *= distortion.getE1() - 1.0 + distortion.getE2() * distortion.getE2();
    result(2, 0) *= distortion.getE1() * distortion.getE2();
    result(0, 1) *= distortion.getE2() * (1.0 + distortion.getE1());
    result(1, 1) *= distortion.getE2() * (1.0 - distortion.getE1());
    result(2, 1) *= 1.0 - distortion.getE1() * distortion.getE1();
    result(0, 2) = 2.0 * ixx;
    result(1, 2) = 2.0 * iyy;
    result(2, 2) = 2.0 * ixy;
    return result;
}

void LogTraceRadius::assignFromQuadrupole(
    double ixx, double iyy, double ixy, 
    Distortion & distortion
) {
    _value = 0.5 * std::log(0.5 * (ixx + iyy));
    distortion.setE1((ixx - iyy) / (ixx + iyy));
    distortion.setE2(2.0 * ixy / (ixx + iyy));
}

BaseCore::Jacobian LogTraceRadius::dAssignFromQuadrupole(
    double ixx, double iyy, double ixy, 
    Distortion & distortion
) {
    double xx_yy = ixx + iyy;
    BaseCore::Jacobian result = BaseCore::Jacobian::Zero();
    _value = 0.5 * std::log(0.5 * xx_yy);
    distortion.setE1((ixx - iyy) / xx_yy);
    distortion.setE2(2.0 * ixy / xx_yy);
    result.block<2,2>(0,0).setConstant(2.0 / (xx_yy * xx_yy));
    result(0, 0) *= iyy;
    result(0, 1) *= -ixx;
    result(1, 0) *= -ixy;
    result(1, 1) *= -ixy;
    result(1, 2) = 2.0 / xx_yy;
    result(2, 0) = 0.5 / xx_yy;
    result(2, 1) = 0.5 / xx_yy;
    return result;
}

void LogTraceRadius::assignToQuadrupole(
    Distortion const & distortion,
    double & ixx, double & iyy, double & ixy
) const {
    double r2 = std::exp(2.0 * _value);
    ixx = r2 * (1.0 + distortion.getE1());
    iyy = r2 * (1.0 - distortion.getE1());
    ixy = r2 * distortion.getE2();
}

BaseCore::Jacobian LogTraceRadius::dAssignToQuadrupole(
    Distortion const & distortion,
    double & ixx, double & iyy, double & ixy
) const {
    BaseCore::Jacobian result = BaseCore::Jacobian::Zero();
    double r2 = std::exp(2.0 * _value);
    ixx = r2 * (1.0 + distortion.getE1());
    iyy = r2 * (1.0 - distortion.getE1());
    ixy = r2 * distortion.getE2();
    result(0, 2) = 2.0 * ixx;
    result(1, 2) = 2.0 * iyy;
    result(2, 2) = 2.0 * ixy;
    result(0, 0) = r2;
    result(1, 0) = -r2;
    result(2, 1) = r2;
    return result;
}

}}}} // namespace lsst::afw::geom::ellipses
