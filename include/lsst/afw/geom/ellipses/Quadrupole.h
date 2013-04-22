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

#ifndef LSST_AFW_GEOM_ELLIPSES_Quadrupole_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_Quadrupole_h_INCLUDED

#include "lsst/afw/geom/ellipses/EllipseCore.h"
#include "lsst/afw/geom/ellipses/Convolution.h"
#include "lsst/afw/geom/ellipses/Transformer.h"
#include "lsst/afw/geom/ellipses/GridTransform.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

/**
 *  @brief An ellipse core with quadrupole moments as parameters.
 *
 *  The quadrupole representation is best thought of as the "covariance matrix" representation
 *  of an ellipse, with parameters corresponding to the three unique elements of the covariance
 *  matrix for an elliptically-symmetric distribution.
 *
 *  The mapping between the quadrupole and other Ellipse does not put any restriction on the
 *  meaning of the ellipse, even though the quadrupole parametrization's naems strongly suggests
 *  a connection with the second moments.  For instance, if you start with an EllipseCore that
 *  corresponds to the half-light radius of an elliptical galaxy model, converting it to a
 *  Quadrupole does not (and cannot) turn this into a measure of its moments - it remains just
 *  another way of parametrizing the half-light radius ellipse.
 */
class Quadrupole : public EllipseCore {
public:

    enum ParameterEnum { IXX=0, IYY=1, IXY=2 }; ///< Enum used to index the elements of a parameter vector.

    /// Matrix type for the matrix representation of Quadrupole parameters.
    typedef Eigen::Matrix<double,2,2,Eigen::DontAlign> Matrix;

    //@{
    /// Basic getters and setters for distinct matrix elements
    double const getIxx() const { return _matrix(0, 0); }
    void setIxx(double ixx) { _matrix(0, 0) = ixx; }

    double const getIyy() const { return _matrix(1, 1); }
    void setIyy(double iyy) { _matrix(1, 1) = iyy; }

    double const getIxy() const { return _matrix(1, 0); }
    void setIxy(double ixy) { _matrix(0, 1) = _matrix(1, 0) = ixy; }
    //@}

    /// @brief Polymorphic deep copy.
    PTR(Quadrupole) clone() const { return boost::static_pointer_cast<Quadrupole>(_clone()); }

    /// Return a string that identifies this parametrization ("Quadrupole").
    virtual std::string getName() const;

    /**
     *  @brief Check parameters and put them into standard form.
     *
     *  In the case of Quadrupole, parameters will never be modified, but they will be checked
     *  for the followig conditions:
     *   - The [0,1] and [1,0] elements of the matrix must be identical.
     *   - Both Ixx and Iyy must be >= 0
     *   - The determinant of the matrix must not be negative.
     *
     *  @throw lsst::pex::exceptions::InvalidParameterException if the above conditions are not met.
     */
    virtual void normalize();

    /// @brief Return a 2x2 symmetric matrix of the parameters.
    Matrix const & getMatrix() const { return _matrix; }

    /// @brief Return the determinant of the matrix representation.
    double getDeterminant() const { return getIxx() * getIyy() - getIxy() * getIxy(); }

    /// @brief Standard assignment.
    Quadrupole & operator=(Quadrupole const & other) { _matrix = other._matrix; return *this; }

    /// @brief Converting assignment.
    Quadrupole & operator=(EllipseCore const & other) { EllipseCore::operator=(other); return *this; }

    /// @brief Construct a circle with the given second moments (Ixx=Iyy=Irr, Ixy=0).
    explicit Quadrupole(double irr=1.0);

    /// @brief Construct from the three matrix elements specified explicitly.
    Quadrupole(double ixx, double iyy, double ixy, bool normalize=false);

    /// @brief Construct from a parameter vector, ordered (Ixx, Iyy, Ixy).
    explicit Quadrupole(EllipseCore::ParameterVector const & vector, bool normalize=false);

    /// @brief Construct from a 2x2 matrix.
    explicit Quadrupole(Matrix const & matrix, bool normalize=true);

    /// @brief Copy constructor.
    Quadrupole(Quadrupole const & other) : _matrix(other._matrix) {}

    /// @brief Converting copy constructor.
    Quadrupole(EllipseCore const & other) { *this = other; }
#ifndef SWIG
    /// @brief Converting copy constructor.
    Quadrupole(EllipseCore::Transformer const & transformer) {
        transformer.apply(*this);
    }

    /// @brief Converting copy constructor.
    Quadrupole(EllipseCore::Convolution const & convolution) {
        convolution.apply(*this);
    }
#endif
protected:

    virtual PTR(EllipseCore) _clone() const { return boost::make_shared<Quadrupole>(*this); }

    virtual void readParameters(double const * iter);
    virtual void writeParameters(double * iter) const;

    virtual void _assignToQuadrupole(double & ixx, double & iyy, double & ixy) const;
    virtual void _assignFromQuadrupole(double ixx, double iyy, double ixy);

    virtual void _assignToAxes(double & a, double & b, double & theta) const;
    virtual void _assignFromAxes(double a, double b, double theta);

    virtual Jacobian _dAssignToQuadrupole(double & ixx, double & iyy, double & ixy) const;
    virtual Jacobian _dAssignFromQuadrupole(double ixx, double iyy, double ixy);

    virtual Jacobian _dAssignToAxes(double & a, double & b, double & theta) const;
    virtual Jacobian _dAssignFromAxes(double a, double b, double theta);

private:
    static Registrar<Quadrupole> registrar;

    Matrix _matrix;
};

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_Quadrupole_h_INCLUDED
