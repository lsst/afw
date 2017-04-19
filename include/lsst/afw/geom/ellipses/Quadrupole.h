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

#ifndef LSST_AFW_GEOM_ELLIPSES_Quadrupole_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_Quadrupole_h_INCLUDED

/*
 *  Definitions and inlines for Quadrupole.
 *
 *  Note: do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/ellipses/BaseCore.h"
#include "lsst/afw/geom/ellipses/Convolution.h"
#include "lsst/afw/geom/ellipses/Transformer.h"
#include "lsst/afw/geom/ellipses/GridTransform.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

/**
 *  An ellipse core with quadrupole moments as parameters.
 */
class Quadrupole : public BaseCore {
public:
    enum ParameterEnum { IXX = 0, IYY = 1, IXY = 2 };  ///< Definitions for elements of a core vector.

    /// Matrix type for the matrix representation of Quadrupole parameters.
    typedef Eigen::Matrix<double, 2, 2, Eigen::DontAlign> Matrix;

    double const getIxx() const { return _matrix(0, 0); }
    void setIxx(double ixx) { _matrix(0, 0) = ixx; }

    double const getIyy() const { return _matrix(1, 1); }
    void setIyy(double iyy) { _matrix(1, 1) = iyy; }

    double const getIxy() const { return _matrix(1, 0); }
    void setIxy(double ixy) { _matrix(0, 1) = _matrix(1, 0) = ixy; }

    /// Deep copy the ellipse core.
    std::shared_ptr<Quadrupole> clone() const { return std::static_pointer_cast<Quadrupole>(_clone()); }

    /// Return a string that identifies this parametrization.
    virtual std::string getName() const;

    /**
     *  @brief Put the parameters into a "standard form", and throw InvalidParameterError
     *         if they cannot be normalized.
     */
    virtual void normalize();

    virtual void readParameters(double const* iter);

    virtual void writeParameters(double* iter) const;

    /// Return a 2x2 symmetric matrix of the parameters.
    Matrix const& getMatrix() const { return _matrix; }

    /// Return the determinant of the matrix representation.
    double getDeterminant() const { return getIxx() * getIyy() - getIxy() * getIxy(); }

    /// Standard assignment.
    Quadrupole& operator=(Quadrupole const& other) {
        _matrix = other._matrix;
        return *this;
    }

    /// Converting assignment.
    Quadrupole& operator=(BaseCore const& other) {
        BaseCore::operator=(other);
        return *this;
    }

    /// Construct from parameter values.
    explicit Quadrupole(double ixx = 1.0, double iyy = 1.0, double ixy = 0.0, bool normalize = false);

    /// Construct from a parameter vector.
    explicit Quadrupole(BaseCore::ParameterVector const& vector, bool normalize = false);

    /// Construct from a 2x2 matrix.
    explicit Quadrupole(Matrix const& matrix, bool normalize = true);

    /// Copy constructor.
    Quadrupole(Quadrupole const& other) : _matrix(other._matrix) {}

    /// Converting copy constructor.
    Quadrupole(BaseCore const& other) { *this = other; }

    /// Converting copy constructor.
    Quadrupole(BaseCore::Transformer const& transformer) { transformer.apply(*this); }

    /// Converting copy constructor.
    Quadrupole(BaseCore::Convolution const& convolution) { convolution.apply(*this); }

protected:
    virtual std::shared_ptr<BaseCore> _clone() const { return std::make_shared<Quadrupole>(*this); }

    virtual void _assignToQuadrupole(double& ixx, double& iyy, double& ixy) const;
    virtual void _assignFromQuadrupole(double ixx, double iyy, double ixy);

    virtual void _assignToAxes(double& a, double& b, double& theta) const;
    virtual void _assignFromAxes(double a, double b, double theta);

    virtual Jacobian _dAssignToQuadrupole(double& ixx, double& iyy, double& ixy) const;
    virtual Jacobian _dAssignFromQuadrupole(double ixx, double iyy, double ixy);

    virtual Jacobian _dAssignToAxes(double& a, double& b, double& theta) const;
    virtual Jacobian _dAssignFromAxes(double a, double b, double theta);

private:
    static Registrar<Quadrupole> registrar;

    Matrix _matrix;
};
}
}
}
}  // namespace lsst::afw::geom::ellipses

#endif  // !LSST_AFW_GEOM_ELLIPSES_Quadrupole_h_INCLUDED
