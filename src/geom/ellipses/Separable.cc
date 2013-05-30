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
#include "lsst/afw/geom/ellipses/Distortion.h"
#include "lsst/afw/geom/ellipses/ConformalShear.h"
#include "lsst/afw/geom/ellipses/ReducedShear.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

template <typename Ellipticity_>
EllipseCore::Registrar< Separable<Ellipticity_> > Separable<Ellipticity_>::registrar;

template <typename Ellipticity_>
std::string Separable<Ellipticity_>::getName() const {
    return Ellipticity_::getName() + "EllipseCore";
}

template <typename Ellipticity_>
void Separable<Ellipticity_>::normalize() {
    _ellipticity.normalize();
    if (!(_radius >= 0.0)) {
        throw LSST_EXCEPT(
            pex::exceptions::InvalidParameterException,
            "Ellipse radius must be >= 0"
        );
    }
}

template <typename Ellipticity_>
void Separable<Ellipticity_>::readParameters(double const * iter) {
    setE1(*iter++);
    setE2(*iter++);
    setRadius(*iter++);
}

template <typename Ellipticity_>
void Separable<Ellipticity_>::writeParameters(double * iter) const {
    *iter++ = getE1();
    *iter++ = getE2();
    *iter++ = getRadius();
}

template <typename Ellipticity_>
void Separable<Ellipticity_>::_stream(std::ostream & os) const {
    os << "(" << Ellipticity::getName() << "(e1=" << getE1() << ", e2=" << getE2() << "), " << getRadius() << ")";
}

template <typename Ellipticity_>
Separable<Ellipticity_> &
Separable<Ellipticity_>::operator=(Separable<Ellipticity_> const & other) {
    _ellipticity = other._ellipticity;
    _radius = other._radius;
    return *this;
}

template <typename Ellipticity_>
Separable<Ellipticity_>::Separable(double radius) :
    _ellipticity(0.0, 0.0), _radius(radius)
{}

template <typename Ellipticity_>
Separable<Ellipticity_>::Separable(double e1, double e2, double radius, bool normalize) :
    _ellipticity(e1, e2), _radius(radius)
{
    if (normalize) this->normalize();
}

template <typename Ellipticity_>
Separable<Ellipticity_>::Separable(
    std::complex<double> const & complex,
    double radius, bool normalize
) : _ellipticity(complex), _radius(radius) {
    if (normalize) this->normalize();
}

template <typename Ellipticity_>
Separable<Ellipticity_>::Separable(Ellipticity const & ellipticity, double radius, bool normalize) :
    _ellipticity(ellipticity), _radius(radius)
{
    if (normalize) this->normalize();
}

template <typename Ellipticity_>
Separable<Ellipticity_>::Separable(EllipseCore::ParameterVector const & vector, bool normalize) :
    _ellipticity(vector[0], vector[1]), _radius(vector[2])
{
    if (normalize) this->normalize();
}

template <typename Ellipticity_>
void Separable<Ellipticity_>::_assignToQuadrupole(double & ixx, double & iyy, double & ixy) const {
    _ellipticity._assignToQuadrupole(_radius, ixx, iyy, ixy);
}

template <typename Ellipticity_>
EllipseCore::Jacobian
Separable<Ellipticity_>::_dAssignToQuadrupole(double & ixx, double & iyy, double & ixy) const {
    return _ellipticity._dAssignToQuadrupole(_radius, ixx, iyy, ixy);
}

template <typename Ellipticity_>
void Separable<Ellipticity_>::_assignToAxes(double & a, double & b, double & theta) const {
    double ixx, iyy, ixy;
    _ellipticity._assignToQuadrupole(_radius, ixx, iyy, ixy);
    _assignQuadrupoleToAxes(ixx, iyy, ixy, a, b, theta);
}

template <typename Ellipticity_>
EllipseCore::Jacobian
Separable<Ellipticity_>::_dAssignToAxes(double & a, double & b, double & theta) const {
    double ixx, iyy, ixy;
    EllipseCore::Jacobian j1 = _ellipticity._dAssignToQuadrupole(_radius, ixx, iyy, ixy);
    EllipseCore::Jacobian j2 = _dAssignQuadrupoleToAxes(ixx, iyy, ixy, a, b, theta);
    return j2 * j1;
}

template <typename Ellipticity_>
void Separable<Ellipticity_>::_assignFromQuadrupole(double ixx, double iyy, double ixy) {
    _ellipticity._assignFromQuadrupole(_radius, ixx, iyy, ixy);
}

template <typename Ellipticity_>
EllipseCore::Jacobian
Separable<Ellipticity_>::_dAssignFromQuadrupole(double ixx, double iyy, double ixy) {
    return _ellipticity._dAssignFromQuadrupole(_radius, ixx, iyy, ixy);
}

template <typename Ellipticity_>
void Separable<Ellipticity_>::_assignFromAxes(double a, double b, double theta) {
    double ixx, iyy, ixy;
    _assignAxesToQuadrupole(a, b, theta, ixx, iyy, ixy);
    _ellipticity._assignFromQuadrupole(_radius, ixx, iyy, ixy);
}

template <typename Ellipticity_>
EllipseCore::Jacobian Separable<Ellipticity_>::_dAssignFromAxes(double a, double b, double theta) {
    double ixx, iyy, ixy;
    EllipseCore::Jacobian j1 = _dAssignAxesToQuadrupole(a, b, theta, ixx, iyy, ixy);
    EllipseCore::Jacobian j2 = _ellipticity._dAssignFromQuadrupole(_radius, ixx, iyy, ixy);
    return j2 * j1;
}

template class Separable<Distortion>;
template class Separable<ConformalShear>;
template class Separable<ReducedShear>;

}}}} // namespace lsst::afw::geom::ellipses
