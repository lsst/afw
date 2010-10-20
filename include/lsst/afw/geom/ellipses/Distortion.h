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
 
#ifndef LSST_AFW_GEOM_ELLIPSES_DISTORTION_H
#define LSST_AFW_GEOM_ELLIPSES_DISTORTION_H

/**
 *  \file
 *  \brief Definitions for Distortion and DistortionEllipse.
 *
 *  \note Do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/ellipses/BaseEllipse.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

/**
 *  \brief An ellipse with a Distortion core.
 *
 *  \ingroup EllipseGroup
 */
class DistortionEllipse : public BaseEllipse {
public:

    typedef boost::shared_ptr<DistortionEllipse> Ptr;
    typedef boost::shared_ptr<DistortionEllipse const> ConstPtr;

    typedef DistortionEllipse Ellipse;
    typedef Distortion Core;

    /// Definitions for elements of an ellipse vector.
    enum Parameters { X=0, Y=1, E1=2, E2=3, R=4 };

    /// \brief Deep-copy the ellipse.
    Ptr clone() const { return Ptr(static_cast<DistortionEllipse*>(_clone()));  }

    /// \brief Return the Core object.
    inline Distortion const & getCore() const;

    /// \brief Return the Core object.
    inline Distortion & getCore();

    /**
     *  \brief Set the parameters of this ellipse from another.
     *
     *  This does not change the parametrization of the ellipse.
     */
    DistortionEllipse & operator=(BaseEllipse const & other) {
        return static_cast<DistortionEllipse &>(BaseEllipse::operator=(other));
    }
    DistortionEllipse & operator=(DistortionEllipse const & other) {
        return static_cast<DistortionEllipse &>(BaseEllipse::operator=(other));
    }

    /// \brief Construct from a PointD and zero-size Core.
    explicit inline DistortionEllipse(PointD const & center = PointD());

    /// \brief Construct from a copy of an Distortion core.
    explicit inline DistortionEllipse(Distortion const & core, PointD const & center = PointD());

    /// \brief Construct from a 5-element parameter vector.
    explicit DistortionEllipse(BaseEllipse::ParameterVector const & vector, bool doNormalize=true);

    /// \brief Converting copy constructor.
    inline DistortionEllipse(BaseEllipse const & other);

    /// \brief Copy constructor.
    inline DistortionEllipse(DistortionEllipse const & other);

protected:
    virtual DistortionEllipse * _clone() const { return new DistortionEllipse(*this); }
};

/**
 *  \brief An ellipse core for complex ellipticity and geometric mean radius.
 *
 *  The complex ellipticity is defined here as
 *  \f[
 *    \delta = \frac{1-q^2}{1+q^2}\exp(2 i \theta)
 *  \f]
 *  where \f$q\f$ is the axis ratio and \f$\theta\f$ is the position angle.  The radius
 *  parameter is defined as the geometric mean of the semimajor and semiminor axes.
 *
 *  \note The class is called "Distortion" instead of "Ellipticity" because, while less common,
 *  "distortion" refers only to this mathematical definition; several definitions of "ellipticity"
 *  appear in astronomical literature.
 *
 *  \ingroup EllipseGroup
 */
class Distortion : public BaseCore {
public:

    typedef boost::shared_ptr<Distortion> Ptr;
    typedef boost::shared_ptr<Distortion const> ConstPtr;

    typedef DistortionEllipse Ellipse;
    typedef Distortion Core;

    enum Parameters { E1=0, E2=1, R=2 }; ///< Definitions for elements of a core vector.

    /// \brief Deep copy the ellipse core.
    Ptr clone() const { return Ptr(_clone()); }

    /// \brief Construct an Ellipse of the appropriate subclass from this and the given center.
    DistortionEllipse::Ptr makeEllipse(PointD const & center = PointD()) const {
        return DistortionEllipse::Ptr(_makeEllipse(center));
    }

    /// \brief Assign other to this and return the derivative of the conversion, d(this)/d(other).
    virtual BaseCore::Jacobian dAssign(BaseCore const & other) {
        return other._dAssignTo(static_cast<Core &>(*this)); 
    }

    /// Return a string that identifies this parametrization.
    virtual char const * getName() const { return "Distortion"; }
    
    /**
     *  \brief Put the parameters into a "standard form", if possible, and return
     *         false if they are entirely invalid.
     */
    virtual bool normalize() { double e = getE(); return e >= 0.0 && e < 1.0 && _vector[R] >= 0; }

    /// \brief Increase the major and minor radii of the ellipse core by the given buffer.
    virtual void scale(double ratio) { _vector[R] *= ratio; }

    /// \brief Return the ellipticity as a complex number.
    std::complex<double> getComplex() const { return std::complex<double>(_vector[E1],_vector[E2]); }

    /// \brief Set the ellipticity from a complex number.
    void setComplex(std::complex<double> const & e) { _vector[E1] = e.real(); _vector[E2] = e.imag(); }

    /// \brief Return the ellipticity magnitude.
    double getE() const { return std::sqrt(_vector[E1]*_vector[E1] + _vector[E2]*_vector[E2]); }

    /// \brief Set the ellipticity magnitude, keeping the position angle constant.
    void setE(double e) { double f = e/getE(); _vector[E1] *= f; _vector[E2] *= f; }

    /// \brief Standard assignment.
    Distortion & operator=(Distortion const & other) { _vector = other._vector; return *this; }

    /// \brief Converting assignment.
    virtual Distortion & operator=(BaseCore const & other) { other._assignTo(*this); return *this; }

    /// \brief Construct from a parameter vector.
    explicit Distortion(BaseCore::ParameterVector const & vector) : BaseCore(vector) {}

    /// \brief Construct from parameter values.
    explicit Distortion(double e1=0, double e2=0, double radius=0) : BaseCore(e1,e2,radius) {}
    
    /// \brief Construct from complex ellipticity and radius.
    explicit Distortion(std::complex<double> const & e, double radius=0) 
        : BaseCore(e.real(),e.imag(),radius) {}

    /// \brief Converting copy constructor.
    Distortion(BaseCore const & other) { *this = other; }

    /// \brief Copy constructor.
    Distortion(Distortion const & other) : BaseCore(other.getVector()) {}

protected:
    
    virtual Distortion * _clone() const { return new Distortion(*this); }
    
    virtual DistortionEllipse * _makeEllipse(PointD const & center) const {
        return new DistortionEllipse(*this, center);
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

inline Distortion const & 
DistortionEllipse::getCore() const { return static_cast<Distortion const &>(*_core); }

inline Distortion & DistortionEllipse::getCore() { return static_cast<Distortion &>(*_core); }

inline DistortionEllipse::DistortionEllipse(PointD const & center) :
    BaseEllipse(new Distortion(),center) {}
inline DistortionEllipse::DistortionEllipse(Distortion const & core, PointD const & center) : 
    BaseEllipse(core,center) {}
inline DistortionEllipse::DistortionEllipse(BaseEllipse const & other) : 
    BaseEllipse(new Distortion(other.getCore()),other.getCenter()) {}
inline DistortionEllipse::DistortionEllipse(DistortionEllipse const & other) : 
    BaseEllipse(other.getCore(),other.getCenter()) {}


} // namespace lsst::afw::geom::ellipses
}}} // namespace lsst::afw::geom
#endif // !LSST_AFW_GEOM_ELLIPSES_DISTORTION_H
