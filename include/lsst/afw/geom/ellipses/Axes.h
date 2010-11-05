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
 
#ifndef LSST_AFW_GEOM_ELLIPSES_AXES_H
#define LSST_AFW_GEOM_ELLIPSES_AXES_H

/**
 *  \file
 *  \brief Definitions for Axes and AxesEllipse.
 *
 *  \note Do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/ellipses/BaseEllipse.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

/**
 *  \brief An ellipse with an Axes core.
 *
 *  \ingroup EllipseGroup
 */
class AxesEllipse : public BaseEllipse {
public:

    typedef boost::shared_ptr<AxesEllipse> Ptr;
    typedef boost::shared_ptr<AxesEllipse const> ConstPtr;

    typedef AxesEllipse Ellipse;
    typedef Axes Core;

    /// Definitions for elements of an ellipse vector.
    enum Parameters { X=0, Y=1, A=2, B=3, THETA=4 };

    /// \brief Deep-copy the ellipse.
    Ptr clone() const { return Ptr(static_cast<AxesEllipse*>(_clone()));  }

    /// \brief Return the Core object.
    inline Axes const & getCore() const;

    /// \brief Return the Core object.
    inline Axes & getCore();

    /**
     *  \brief Set the parameters of this ellipse from another.
     *
     *  This does not change the parametrization of the ellipse.
     */
    AxesEllipse & operator=(BaseEllipse const & other) {
        return static_cast<AxesEllipse &>(BaseEllipse::operator=(other));
    }
    AxesEllipse & operator=(AxesEllipse const & other) {
        return static_cast<AxesEllipse &>(BaseEllipse::operator=(other));
    }

    /// \brief Construct from a PointD and zero-size Core.
    explicit inline AxesEllipse(PointD const & center = PointD());

    /// \brief Construct from a copy of an Axes core.
    explicit inline AxesEllipse(Axes const & core, PointD const & center = PointD());

    /// \brief Construct from a 5-element parameter vector.
    explicit AxesEllipse(BaseEllipse::ParameterVector const & vector, bool doNormalize=true);

    /// \brief Converting copy constructor.
    inline AxesEllipse(BaseEllipse const & other);

    /// \brief Copy constructor.
    inline AxesEllipse(AxesEllipse const & other);

protected:
    virtual AxesEllipse * _clone() const { return new AxesEllipse(*this); }
};

/**
 *  \brief An ellipse core for the semimajor/semiminor axis and position angle parametrization (a,b,theta).
 *
 *  \warning The conversion Jacobians (result of dAssign) between Axes and other types are not
 *           well-defined for exact circles.  To avoid problems, avoid differentiating expressions
 *           involving Axes ellipse cores.
 *
 *  \ingroup EllipseGroup
 */
class Axes : public BaseCore {
public:

    typedef boost::shared_ptr<Axes> Ptr;
    typedef boost::shared_ptr<Axes const> ConstPtr;

    typedef AxesEllipse Ellipse;
    typedef Axes Core;

    enum Parameters { A=0, B=1, THETA=2 }; ///< Definitions for elements of a core vector.

    /// \brief Deep copy the ellipse core.
    Ptr clone() const { return Ptr(_clone()); }

    /// \brief Construct an Ellipse of the appropriate subclass from this and the given center.
    AxesEllipse::Ptr makeEllipse(PointD const & center = PointD()) const {
        return AxesEllipse::Ptr(_makeEllipse(center));
    }

    /// \brief Assign other to this and return the derivative of the conversion, d(this)/d(other).
    virtual BaseCore::Jacobian dAssign(BaseCore const & other) {
        return other._dAssignTo(static_cast<Core &>(*this)); 
    }

    /// Return a string that identifies this parametrization.
    virtual char const * getName() const { return "Axes"; }

    /**
     *  \brief Put the parameters into a "standard form", if possible, and return
     *         false if they are entirely invalid.
     */
    virtual bool normalize(); // swap a,b and rotate if a<b, ensure theta in [-pi/2,pi/2)

    /// \brief Scale the size of the ellipse core by the given factor.
    virtual void grow(double buffer) { _vector[A] += buffer; _vector[B] += buffer; }

    /// \brief Increase the major and minor radii of the ellipse core by the given buffer.
    virtual void scale(double factor) { _vector[A] *= factor; _vector[B] *= factor; }

    /// \brief Return the AffineTransform that transforms the unit circle into this.
    virtual LinearTransform getGenerator() const;

    /// \brief Standard assignment.
    Axes & operator=(Axes const & other) { _vector = other._vector; return *this; }

    /// \brief Converting assignment.
    virtual Axes & operator=(BaseCore const & other) { other._assignTo(*this); return *this; }

    /// \brief Construct from a parameter vector.
    explicit Axes(BaseCore::ParameterVector const & data, bool doNormalize=true) : 
        BaseCore(data) { if (doNormalize) normalize(); }

    /// \brief Construct from parameter values.
    explicit Axes(double a=0, double b=0, double theta=0, bool doNormalize=true) : 
        BaseCore(a,b,theta) { if (doNormalize) normalize(); }

    /// \brief Converting copy constructor.
    Axes(BaseCore const & other) { *this = other; }

    /// \brief Copy constructor.
    Axes(Axes const & other) : BaseCore(other.getVector()) {}

protected:

    virtual Axes * _clone() const { return new Axes(*this); }
    
    virtual AxesEllipse * _makeEllipse(PointD const & center) const {
        return new AxesEllipse(*this, center);
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

inline Axes const & AxesEllipse::getCore() const { return static_cast<Axes const &>(*_core); }
inline Axes & AxesEllipse::getCore() { return static_cast<Axes &>(*_core); }

inline AxesEllipse::AxesEllipse(PointD const & center) :
    BaseEllipse(new Axes(),center) {}
inline AxesEllipse::AxesEllipse(Axes const & core, PointD const & center) : 
    BaseEllipse(core,center) {}
inline AxesEllipse::AxesEllipse(BaseEllipse const & other) : 
    BaseEllipse(new Axes(other.getCore()),other.getCenter()) {}
inline AxesEllipse::AxesEllipse(AxesEllipse const & other) : 
    BaseEllipse(other.getCore(),other.getCenter()) {}

} // namespace lsst::afw::geom::ellipses
}}} // namespace lsst::afw::geom
#endif // !LSST_AFW_GEOM_ELLIPSES_AXES_H
