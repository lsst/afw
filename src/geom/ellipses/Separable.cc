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
#include "lsst/afw/geom/ellipses/Separable.h"
#include "lsst/afw/geom/ellipses/radii.h"
#include "lsst/afw/geom/ellipses/Distortion.h"
#include "lsst/afw/geom/ellipses/ConformalShear.h"
#include "lsst/afw/geom/ellipses/ReducedShear.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

template <typename Ellipticity_, typename Radius_>
EllipseCore::Registrar< Separable<Ellipticity_,Radius_> > Separable<Ellipticity_,Radius_>::registrar;

template <typename Ellipticity_, typename Radius_>
std::string Separable<Ellipticity_,Radius_>::getName() const {
    return "Separable" + Ellipticity_::getName() + Radius_::getName();
}

template <typename Ellipticity_, typename Radius_>
void Separable<Ellipticity_,Radius_>::normalize() {
    _ellipticity.normalize();
    _radius.normalize();
}

template <typename Ellipticity_, typename Radius_>
void Separable<Ellipticity_,Radius_>::readParameters(double const * iter) {
    setE1(*iter++);
    setE2(*iter++);
    setRadius(*iter++);
}

template <typename Ellipticity_, typename Radius_>
void Separable<Ellipticity_,Radius_>::writeParameters(double * iter) const {
    *iter++ = getE1();
    *iter++ = getE2();
    *iter++ = getRadius();
}

template <typename Ellipticity_, typename Radius_>
void Separable<Ellipticity_,Radius_>::_stream(std::ostream & os) const {
    os << "(" << Ellipticity::getName() << "(e1=" << getE1() << ", e2=" << getE2() << "), "
       << Radius::getName() << "(" << double(getRadius()) << ")";
}

template <typename Ellipticity_, typename Radius_>
Separable<Ellipticity_,Radius_> &
Separable<Ellipticity_,Radius_>::operator=(Separable<Ellipticity_,Radius_> const & other) {
    _ellipticity = other._ellipticity;
    _radius = other._radius;
    return *this;
}

template <typename Ellipticity_, typename Radius_>
Separable<Ellipticity_,Radius_>::Separable(double radius) :
    _ellipticity(0.0, 0.0), _radius(radius)
{}

template <typename Ellipticity_, typename Radius_>
Separable<Ellipticity_,Radius_>::Separable(double e1, double e2, double radius, bool normalize) :
    _ellipticity(e1, e2), _radius(radius)
{
    if (normalize) this->normalize();
}

template <typename Ellipticity_, typename Radius_>
Separable<Ellipticity_,Radius_>::Separable(
    std::complex<double> const & complex,
    double radius, bool normalize
) : _ellipticity(complex), _radius(radius) {
    if (normalize) this->normalize();
}

template <typename Ellipticity_, typename Radius_>
Separable<Ellipticity_,Radius_>::Separable(Ellipticity const & ellipticity, double radius, bool normalize) :
    _ellipticity(ellipticity), _radius(radius)
{
    if (normalize) this->normalize();
}

template <typename Ellipticity_, typename Radius_>
Separable<Ellipticity_,Radius_>::Separable(EllipseCore::ParameterVector const & vector, bool normalize) :
    _ellipticity(vector[0], vector[1]), _radius(vector[2])
{
    if (normalize) this->normalize();
}

template <typename Ellipticity_, typename Radius_>
void Separable<Ellipticity_,Radius_>::_assignToQuadrupole(double & ixx, double & iyy, double & ixy) const {
    Distortion distortion(_ellipticity);
    _radius.assignToQuadrupole(distortion, ixx, iyy, ixy);
}

template <typename Ellipticity_, typename Radius_>
EllipseCore::Jacobian
Separable<Ellipticity_,Radius_>::_dAssignToQuadrupole(double & ixx, double & iyy, double & ixy) const {
    Distortion distortion;
    EllipseCore::Jacobian rhs = Jacobian::Identity();
    rhs.block<2,2>(0,0) = distortion.dAssign(_ellipticity);
    EllipseCore::Jacobian lhs = _radius.dAssignToQuadrupole(distortion, ixx, iyy, ixy);
    return lhs * rhs;
}

template <typename Ellipticity_, typename Radius_>
void Separable<Ellipticity_,Radius_>::_assignToAxes(double & a, double & b, double & theta) const {
    double ixx, iyy, ixy;
    this->_assignToQuadrupole(ixx, iyy, ixy);
    EllipseCore::_assignQuadrupoleToAxes(ixx, iyy, ixy, a, b, theta);
}

template <typename Ellipticity_, typename Radius_>
EllipseCore::Jacobian
Separable<Ellipticity_,Radius_>::_dAssignToAxes(double & a, double & b, double & theta) const {
    double ixx, iyy, ixy;
    EllipseCore::Jacobian rhs = this->_dAssignToQuadrupole(ixx, iyy, ixy);
    EllipseCore::Jacobian lhs = EllipseCore::_dAssignQuadrupoleToAxes(ixx, iyy, ixy, a, b, theta);
    return lhs * rhs;
}

template <typename Ellipticity_, typename Radius_>
void Separable<Ellipticity_,Radius_>::_assignFromQuadrupole(double ixx, double iyy, double ixy) {
    Distortion distortion;
    _radius.assignFromQuadrupole(ixx, iyy, ixy, distortion);
    _ellipticity = distortion;
}

template <typename Ellipticity_, typename Radius_>
EllipseCore::Jacobian
Separable<Ellipticity_,Radius_>::_dAssignFromQuadrupole(double ixx, double iyy, double ixy) {
    Distortion distortion;
    EllipseCore::Jacobian rhs = _radius.dAssignFromQuadrupole(ixx, iyy, ixy, distortion);
    EllipseCore::Jacobian lhs = EllipseCore::Jacobian::Identity();
    lhs.block<2,2>(0,0) = _ellipticity.dAssign(distortion);
    return lhs * rhs;
}

template <typename Ellipticity_, typename Radius_>
void Separable<Ellipticity_,Radius_>::_assignFromAxes(double a, double b, double theta) {
    double ixx, iyy, ixy;
    EllipseCore::_assignAxesToQuadrupole(a, b, theta, ixx, iyy, ixy);
    this->_assignFromQuadrupole(ixx, iyy, ixy);
}

template <typename Ellipticity_, typename Radius_>
EllipseCore::Jacobian Separable<Ellipticity_,Radius_>::_dAssignFromAxes(double a, double b, double theta) {
    double ixx, iyy, ixy;
    EllipseCore::Jacobian rhs = EllipseCore::_dAssignAxesToQuadrupole(a, b, theta, ixx, iyy, ixy);
    EllipseCore::Jacobian lhs = this->_dAssignFromQuadrupole(ixx, iyy, ixy);
    return lhs * rhs;
}

template class Separable<Distortion,DeterminantRadius>;
template class Separable<Distortion,TraceRadius>;
template class Separable<Distortion,LogDeterminantRadius>;
template class Separable<Distortion,LogTraceRadius>;

template class Separable<ConformalShear,DeterminantRadius>;
template class Separable<ConformalShear,TraceRadius>;
template class Separable<ConformalShear,LogDeterminantRadius>;
template class Separable<ConformalShear,LogTraceRadius>;

template class Separable<ReducedShear,DeterminantRadius>;
template class Separable<ReducedShear,TraceRadius>;
template class Separable<ReducedShear,LogDeterminantRadius>;
template class Separable<ReducedShear,LogTraceRadius>;

}}}} // namespace lsst::afw::geom::ellipses
