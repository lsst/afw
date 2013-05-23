// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2013 LSST Corporation.
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

#include <map>

#include <boost/format.hpp>

#include "lsst/afw/geom/ellipses/EllipseCore.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/afw/geom/ellipses/Axes.h"
#include "lsst/afw/geom/Angle.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

namespace {

typedef std::map< std::string, boost::shared_ptr<EllipseCore> > RegistryMap;

RegistryMap & getRegistry() {
    static RegistryMap instance;
    return instance;
}

PTR(EllipseCore) getRegistryCopy(std::string const & name) {
    RegistryMap::iterator i = getRegistry().find(name);
    if (i == getRegistry().end()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Ellipse core with name '%s' not found in registry.") % name).str()
        );
    }
    return i->second->clone();
}

} // anonymous

PTR(EllipseCore) EllipseCore::make(std::string const & name) {
    PTR(EllipseCore) result = getRegistryCopy(name);
    *result = Quadrupole();
    return result;
}

PTR(EllipseCore) EllipseCore::make(std::string const & name, ParameterVector const & parameters) {
    PTR(EllipseCore) result = getRegistryCopy(name);
    result->setParameterVector(parameters);
    return result;
}

PTR(EllipseCore) EllipseCore::make(std::string const & name, double v1, double v2, double v3) {
    PTR(EllipseCore) result = getRegistryCopy(name);
    result->setParameterVector(ParameterVector(v1, v2, v3));
    return result;
}

PTR(EllipseCore) EllipseCore::make(std::string const & name, EllipseCore const & other) {
    PTR(EllipseCore) result = getRegistryCopy(name);
    *result = other;
    return result;
}

PTR(EllipseCore) EllipseCore::make(std::string const & name, Transformer const & other) {
    PTR(EllipseCore) result = getRegistryCopy(name);
    other.apply(*result);
    return result;
}

PTR(EllipseCore) EllipseCore::make(std::string const & name, Convolution const & other) {
    PTR(EllipseCore) result = getRegistryCopy(name);
    other.apply(*result);
    return result;
}

void EllipseCore::registerSubclass(PTR(EllipseCore) const & example) {
    getRegistry()[example->getName()] = example;
}

void EllipseCore::grow(double buffer) {
    double a, b, theta;
    _assignToAxes(a, b, theta);
    a += buffer;
    b += buffer;
    _assignFromAxes(a, b, theta);
}

void EllipseCore::scale(double factor) {
    double a, b, theta;
    _assignToAxes(a, b, theta);
    a *= factor;
    b *= factor;
    _assignFromAxes(a, b, theta);
}

double EllipseCore::getArea() const {
    double a, b, theta;
    _assignToAxes(a, b, theta);
    return a * b * geom::PI;
}

double EllipseCore::getDeterminantRadius() const {
    double a, b, theta;
    _assignToAxes(a, b, theta);
    return std::sqrt(a * b);
}

double EllipseCore::getTraceRadius() const {
    double ixx, iyy, ixy;
    _assignToQuadrupole(ixx, iyy, ixy);
    return std::sqrt(0.5 * (ixx + iyy));
}

Extent2D EllipseCore::computeDimensions() const {
    double a, b, theta;
    _assignToAxes(a, b, theta);
    double c = std::cos(theta);
    double s = std::sin(theta);
    c *= c;
    s *= s;
    b *= b;
    a *= a;
    Extent2D dimensions(std::sqrt(b * s + a * c),std::sqrt(a * s + b * c));
    dimensions *= 2;
    return dimensions;
}

EllipseCore::ParameterVector const EllipseCore::getParameterVector() const {
    ParameterVector r;
    writeParameters(r.data());
    return r;
}

void EllipseCore::setParameterVector(ParameterVector const & p) {
    readParameters(p.data());
}

bool EllipseCore::operator==(EllipseCore const & other) const {
    return getParameterVector() == other.getParameterVector() && getName() == other.getName();
}

EllipseCore & EllipseCore::operator=(EllipseCore const & other) {
    if (&other != this) {
        // We use Axes instead of Quadrupole here because it allows us to copy Axes without
        // implicitly normalizing them.
        double a, b, theta;
        other._assignToAxes(a, b, theta);
        _assignFromAxes(a, b, theta);
    }
    return *this;
}

EllipseCore::Jacobian EllipseCore::dAssign(EllipseCore const & other) {
    if (getName() == other.getName()) {
        this->operator=(other);
        return Jacobian::Identity();
    }
    // We use Quadrupole instead of Axes here because the ambiguity of the position angle
    // in the circular case causes some of the Jacobians to/from Axes to be undefined for
    // exact circles.  Quadrupoles don't have that problem, and the Axes-to-Axes case is
    // handled by the above if block.
    double ixx, iyy, ixy;
    Jacobian rhs = other._dAssignToQuadrupole(ixx, iyy, ixy);
    Jacobian lhs = _dAssignFromQuadrupole(ixx, iyy, ixy);
    return lhs * rhs;
}

void EllipseCore::_assignQuadrupoleToAxes(
    double ixx, double iyy, double ixy,
    double & a, double & b, double & theta
) {
    double xx_p_yy = ixx + iyy;
    double xx_m_yy = ixx - iyy;
    double t = std::sqrt(xx_m_yy*xx_m_yy + 4*ixy*ixy);
    a = std::sqrt(0.5*(xx_p_yy + t));
    b = std::sqrt(0.5*(xx_p_yy - t));
    theta = 0.5*std::atan2(2.0*ixy, xx_m_yy);
}

EllipseCore::Jacobian EllipseCore::_dAssignQuadrupoleToAxes(
    double ixx, double iyy, double ixy,
    double & a, double & b, double & theta
) {
    double xx_p_yy = ixx + iyy;
    double xx_m_yy = ixx - iyy;
    double t2 = xx_m_yy*xx_m_yy + 4.0*ixy*ixy;
    Eigen::Vector3d dt2(2.0*xx_m_yy, -2.0*xx_m_yy, 8.0*ixy);
    double t = std::sqrt(t2);
    a = std::sqrt(0.5*(xx_p_yy + t));
    b = std::sqrt(0.5*(xx_p_yy - t));
    theta = 0.5*std::atan2(2.0*ixy, xx_m_yy);
    Jacobian m = Jacobian::Zero();
    m(0, 0) = 0.25 * (1.0 + 0.5 * dt2[0] / t) / a;
    m(0, 1) = 0.25 * (1.0 + 0.5 * dt2[1] / t) / a;
    m(0, 2) = 0.25 * (0.5 * dt2[2] / t) / a;
    m(1, 0) = 0.25 * (1.0 - 0.5 * dt2[0] / t) / b;
    m(1, 1) = 0.25 * (1.0 - 0.5 * dt2[1] / t) / b;
    m(1, 2) = 0.25 * (-0.5 * dt2[2] / t) / b;

    m.row(2).setConstant(1.0 / (t * t));
    m(2, 0) *= -ixy;
    m(2, 1) *= ixy;
    m(2, 2) *= xx_m_yy;
    return m;
}

void EllipseCore::_assignAxesToQuadrupole(
    double a, double b, double theta,
    double & ixx, double & iyy, double & ixy
) {
    a *= a;
    b *= b;
    double c = std::cos(theta);
    double s = std::sin(theta);
    ixy = (a - b)*c*s;
    c *= c;
    s *= s;
    ixx = c*a + s*b;
    iyy = s*a + c*b;
}

EllipseCore::Jacobian EllipseCore::_dAssignAxesToQuadrupole(
    double a, double b, double theta,
    double & ixx, double & iyy, double & ixy
) {
    Jacobian m;
    m.col(0).setConstant(2*a);
    m.col(1).setConstant(2*b);
    a *= a;
    b *= b;
    m.col(2).setConstant(a-b);
    double c = std::cos(theta);
    double s = std::sin(theta);
    double cs = c*s;
    ixy = (a - b)*c*s;
    c *= c;
    s *= s;
    ixx = c*a + s*b;
    iyy = s*a + c*b;
    m(0,0) *= c;  m(0,1) *= s;   m(0,2) *= -2.0*cs;
    m(1,0) *= s;  m(1,1) *= c;   m(1,2) *= 2.0*cs;
    m(2,0) *= cs; m(2,1) *= -cs; m(2,2) *= (c - s);
    return m;
}

}}}} // namespace lsst::afw::geom::ellipses
