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
 
#ifndef LSST_AFW_GEOM_ELLIPSES_QUADRUPOLE_H
#define LSST_AFW_GEOM_ELLIPSES_QUADRUPOLE_H

/**
 *  \file
 *  \brief Definitions for Quadrupole and QuadrupoleEllipse.
 *
 *  \note Do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/ellipses/BaseEllipse.h"

namespace lsst {
namespace afw {
namespace geom { 
namespace ellipses {

/**
 *  \brief An ellipse with a Quadrupole core.
 *
 *  \ingroup EllipseGroup
 */
class QuadrupoleEllipse : public BaseEllipse {
public:

    typedef boost::shared_ptr<QuadrupoleEllipse> Ptr;
    typedef boost::shared_ptr<QuadrupoleEllipse const> ConstPtr;

    typedef QuadrupoleEllipse Ellipse;
    typedef Quadrupole Core;

    /// Definitions for elements of an ellipse vector.
    enum Parameters { X=0, Y=1, IXX=2, IYY=3, IXY=4 };

    /// \brief Deep-copy the ellipse.
    Ptr clone() const { return Ptr(static_cast<QuadrupoleEllipse*>(_clone()));  }

    /// \brief Return the Core object.
    inline Quadrupole const & getCore() const;

    /// \brief Return the Core object.
    inline Quadrupole & getCore();

    /**
     *  \brief Set the parameters of this ellipse from another.
     *
     *  This does not change the parametrization of the ellipse.
     */
    QuadrupoleEllipse & operator=(BaseEllipse const & other) {
        return static_cast<QuadrupoleEllipse &>(BaseEllipse::operator=(other)); 
    }
    QuadrupoleEllipse & operator=(QuadrupoleEllipse const & other) {
        return static_cast<QuadrupoleEllipse &>(BaseEllipse::operator=(other)); 
    }

    /// \brief Construct from a PointD and zero-size Core.
    explicit inline QuadrupoleEllipse(PointD const & center = PointD());

    /// \brief Construct from a copy of an Quadrupole core.
    explicit inline QuadrupoleEllipse(Quadrupole const & core, PointD const & center = PointD());

    /// \brief Construct from a 5-element parameter vector.
    explicit QuadrupoleEllipse(BaseEllipse::ParameterVector const & vector, bool doNormalize=true);

    /// \brief Converting copy constructor.
    inline QuadrupoleEllipse(BaseEllipse const & other);

    /// \brief Copy constructor.
    inline QuadrupoleEllipse(QuadrupoleEllipse const & other);

protected:
    virtual QuadrupoleEllipse * _clone() const { return new QuadrupoleEllipse(*this); }
};

/**
 *  \brief An ellipse core using the quadrupole moments of an elliptical Gaussian (ixx,iyy,ixy) 
 *         as its parameters.
 *
 *  \ingroup EllipseGroup
 */
class Quadrupole : public BaseCore {
public:

    typedef boost::shared_ptr<Quadrupole> Ptr;
    typedef boost::shared_ptr<Quadrupole const> ConstPtr;

    typedef QuadrupoleEllipse Ellipse;
    typedef Quadrupole Core;

    typedef Eigen::Matrix2d Matrix; ///< Matrix type for the matrix representation of Quadrupole parameters.

    enum Parameters { IXX=0, IYY=1, IXY=2 }; ///< Definitions for elements of a core vector.

    /// \brief Deep copy the ellipse core.
    Ptr clone() const { return Ptr(_clone()); }

    /// \brief Construct an Ellipse of the appropriate subclass from this and the given center.
    QuadrupoleEllipse::Ptr makeEllipse(PointD const & center = PointD()) const {
        return QuadrupoleEllipse::Ptr(_makeEllipse(center));
    }

    /// \brief Assign other to this and return the derivative of the conversion, d(this)/d(other).
    virtual BaseCore::Jacobian dAssign(BaseCore const & other) {
        return other._dAssignTo(static_cast<Core &>(*this)); 
    }

    /// Return a string that identifies this parametrization.
    virtual char const * getName() const { return "Quadrupole"; }

    /**
     *  \brief Put the parameters into a "standard form", if possible, and return
     *         false if they are entirely invalid.
     */
    virtual bool normalize() { return getDeterminant() >= 0; }

    /// \brief Increase the major and minor radii of the ellipse core by the given buffer.
    virtual void scale(double ratio) { _vector *= (ratio*ratio); }

    /// \brief Return a 2x2 symmetric matrix of the parameters.
    Matrix getMatrix() const {
        Matrix r; r << _vector[IXX], _vector[IXY], _vector[IXY], _vector[IYY]; return r;
    }

    /// \brief Return the determinant of the matrix representation.
    double getDeterminant() const { return _vector[IXX]*_vector[IYY] - _vector[IXY]*_vector[IXY]; }

    /// \brief Standard assignment.
    Quadrupole & operator=(Quadrupole const & other) { _vector = other._vector; return *this; }

    /// \brief Converting assignment.
    virtual Quadrupole & operator=(BaseCore const & other) { other._assignTo(*this); return *this; }

    /// \brief Construct from a parameter vector.
    explicit Quadrupole(BaseCore::ParameterVector const & vector) : BaseCore(vector) {}

    /// \brief Construct from parameter values.
    explicit Quadrupole(double xx=0, double yy=0, double xy=0) : BaseCore(xx,yy,xy) {}
    
    /// \brief Converting copy constructor.
    Quadrupole(BaseCore const & other) { *this = other; }

    /// \brief Copy constructor.
    Quadrupole(Quadrupole const & other) : BaseCore(other.getVector()) {}

protected:
    
    virtual Quadrupole * _clone() const { return new Quadrupole(*this); }
    
    virtual QuadrupoleEllipse * _makeEllipse(PointD const & center) const {
        return new QuadrupoleEllipse(*this, center);
    }

    virtual void _assignTo(Quadrupole & other) const;
    virtual void _assignTo(Axes & other) const;
    virtual void _assignTo(Distortion & other) const;
    virtual void _assignTo(LogShear & other) const;

    virtual Jacobian _dAssignTo(Quadrupole & other) const;
    virtual Jacobian _dAssignTo(Axes & other) const;
    virtual Jacobian _dAssignTo(Distortion & other) const;
    virtual Jacobian _dAssignTo(LogShear & other) const;

};

inline Quadrupole const & 
QuadrupoleEllipse::getCore() const { return static_cast<Quadrupole const &>(*_core); }

inline Quadrupole & QuadrupoleEllipse::getCore() { return static_cast<Quadrupole &>(*_core); }

inline QuadrupoleEllipse::QuadrupoleEllipse(PointD const & center) :
    BaseEllipse(new Quadrupole(),center) {}
inline QuadrupoleEllipse::QuadrupoleEllipse(Quadrupole const & core, PointD const & center) : 
    BaseEllipse(core,center) {}
inline QuadrupoleEllipse::QuadrupoleEllipse(BaseEllipse const & other) : 
    BaseEllipse(other.getCore(),other.getCenter()) {}
inline QuadrupoleEllipse::QuadrupoleEllipse(QuadrupoleEllipse const & other) : 
    BaseEllipse(other.getCore(),other.getCenter()) {}

} // namespace lsst::afw::geom::ellipses
}}} // namespace lsst::afw::geom
#endif // !LSST_AFW_GEOM_ELLIPSES_QUADRUPOLE_H
