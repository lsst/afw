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
#include "lsst/afw/geom/ellipses/Axes.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

BaseCore::Registrar<Quadrupole> Quadrupole::registrar;

std::string Quadrupole::getName() const { return "Quadrupole"; }

void Quadrupole::normalize() {
    if (_matrix(0, 1) != _matrix(1, 0))
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "Quadrupole matrix must be symmetric.");
    if (getIxx() < 0 || getIyy() < 0)
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "Quadrupole matrix cannot have negative diagonal elements.");
    if (getDeterminant() < 0)
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "Quadrupole matrix cannot have negative determinant.");
}

void Quadrupole::readParameters(double const* iter) {
    setIxx(*iter++);
    setIyy(*iter++);
    setIxy(*iter++);
}

void Quadrupole::writeParameters(double* iter) const {
    *iter++ = getIxx();
    *iter++ = getIyy();
    *iter++ = getIxy();
}

Quadrupole::Quadrupole(double ixx, double iyy, double ixy, bool normalize) {
    setIxx(ixx);
    setIyy(iyy);
    setIxy(ixy);
    if (normalize) this->normalize();
}

Quadrupole::Quadrupole(BaseCore::ParameterVector const& vector, bool normalize) {
    setIxx(vector[IXX]);
    setIyy(vector[IYY]);
    setIxy(vector[IXY]);
    if (normalize) this->normalize();
}

Quadrupole::Quadrupole(Matrix const& matrix, bool normalize) : _matrix(matrix) {
    if (normalize) this->normalize();
}

void Quadrupole::_assignToQuadrupole(double& ixx, double& iyy, double& ixy) const {
    ixx = getIxx();
    iyy = getIyy();
    ixy = getIxy();
}

BaseCore::Jacobian Quadrupole::_dAssignToQuadrupole(double& ixx, double& iyy, double& ixy) const {
    ixx = getIxx();
    iyy = getIyy();
    ixy = getIxy();
    return Jacobian::Identity();
}

void Quadrupole::_assignToAxes(double& a, double& b, double& theta) const {
    BaseCore::_assignQuadrupoleToAxes(getIxx(), getIyy(), getIxy(), a, b, theta);
}

BaseCore::Jacobian Quadrupole::_dAssignToAxes(double& a, double& b, double& theta) const {
    return BaseCore::_dAssignQuadrupoleToAxes(getIxx(), getIyy(), getIxy(), a, b, theta);
}

void Quadrupole::_assignFromQuadrupole(double ixx, double iyy, double ixy) {
    setIxx(ixx);
    setIyy(iyy);
    setIxy(ixy);
}

BaseCore::Jacobian Quadrupole::_dAssignFromQuadrupole(double ixx, double iyy, double ixy) {
    setIxx(ixx);
    setIyy(iyy);
    setIxy(ixy);
    return Jacobian::Identity();
}

void Quadrupole::_assignFromAxes(double a, double b, double theta) {
    BaseCore::_assignAxesToQuadrupole(a, b, theta, _matrix(0, 0), _matrix(1, 1), _matrix(0, 1));
    _matrix(1, 0) = _matrix(0, 1);
}

BaseCore::Jacobian Quadrupole::_dAssignFromAxes(double a, double b, double theta) {
    Jacobian r = BaseCore::_dAssignAxesToQuadrupole(a, b, theta, _matrix(0, 0), _matrix(1, 1), _matrix(0, 1));
    _matrix(1, 0) = _matrix(0, 1);
    return r;
}
}
}
}
}  // namespace lsst::afw::geom::ellipses
