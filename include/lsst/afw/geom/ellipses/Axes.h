// -*- lsst-c++ -*-
#ifndef LSST_AFW_GEOM_ELLIPSES_AXES_H
#define LSST_AFW_GEOM_ELLIPSES_AXES_H

/**
 *  \file
 *  \brief Definitions for Axes and AxesEllipse.
 *
 *  \note Do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/ellipses/EllipseImpl.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

class AxesEllipse;

/**
 *  \brief An ellipse core for the semimajor/semiminor axis and position angle parametrization (a,b,theta).
 *
 *  \ingroup EllipseGroup
 */
class Axes : public detail::CoreImpl<Axes,AxesEllipse> {
    typedef detail::CoreImpl<Axes,AxesEllipse> Super;
public:

    enum Parameters { A=0, B=1, THETA=2 }; ///< Definitions for elements of a core vector.

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
    virtual AffineTransform getGenerator() const;

    /// \brief Standard assignment.
    Axes & operator=(Axes const & other) { _vector = other._vector; return *this; }

    /// \brief Converting assignment.
    virtual Axes & operator=(BaseCore const & other) { other._assignTo(*this); return *this; }

    /// \brief Construct from a parameter vector.
    explicit Axes(ParameterVector const & data, bool doNormalize=true) : 
        Super(data) { if (doNormalize) normalize(); }

    /// \brief Construct from parameter values.
    explicit Axes(double a=0, double b=0, double theta=0, bool doNormalize=true) : 
        Super(a,b,theta) { if (doNormalize) normalize(); }

    /// \brief Converting copy constructor.
    Axes(BaseCore const & other) { *this = other; }

    /// \brief Copy constructor.
    Axes(Axes const & other) : Super(other.getVector()) {}

protected:

    virtual void _assignTo(Quadrupole & other) const;
    virtual void _assignTo(Axes & other) const;
    virtual void _assignTo(Distortion & other) const;
    virtual void _assignTo(LogShear & other) const;

    virtual Jacobian _dAssignTo(Quadrupole & other) const;
    virtual Jacobian _dAssignTo(Axes & other) const;
    virtual Jacobian _dAssignTo(Distortion & other) const;
    virtual Jacobian _dAssignTo(LogShear & other) const;

};

/**
 *  \brief An ellipse with an Axes core.
 *
 *  \ingroup EllipseGroup
 */
class AxesEllipse : public detail::EllipseImpl<Axes,AxesEllipse> {
    typedef detail::EllipseImpl<Axes,AxesEllipse> Super;
public:

    /// Definitions for elements of an ellipse vector.
    enum Parameters { X=0, Y=1, A=2, B=3, THETA=4 };

    /**
     *  \brief Set the parameters of this ellipse from another.
     *
     *  This does not change the parametrization of the ellipse.
     */
    AxesEllipse & operator=(BaseEllipse const & other) { return Super::operator=(other); }

    /// \brief Set the parameters of this ellipse from another.
    AxesEllipse & operator=(AxesEllipse const & other) { return Super::operator=(other); }

    /// \brief Construct from a PointD and zero-size Core.
    explicit AxesEllipse(PointD const & center = PointD()) : Super(new Axes(),center) {}

    /// \brief Construct from a copy of an Axes core.
    explicit AxesEllipse(Axes const & core, PointD const & center = PointD()) : Super(core,center) {}

    /// \brief Construct from a pointer to an Axes core.
    explicit AxesEllipse(std::auto_ptr<Axes> core, PointD const & center = PointD()) : 
        Super(core.release(),center) {}

    /// \brief Construct from a 5-element parameter vector.
    explicit AxesEllipse(BaseEllipse::ParameterVector const & vector, bool doNormalize=true) :
        Super(vector) { if (doNormalize) normalize(); }

    /// \brief Converting copy constructor.
    AxesEllipse(BaseEllipse const & other) : Super(other.getCore(),other.getCenter()) {}

    /// \brief Copy constructor.
    AxesEllipse(AxesEllipse const & other) : Super(other.getCore(),other.getCenter()) {}

};

} // namespace lsst::afw::geom::ellipses
}}} // namespace lsst::afw::geom
#endif // !LSST_AFW_GEOM_ELLIPSES_AXES_H
