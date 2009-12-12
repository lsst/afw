// -*- lsst-c++ -*-
#ifndef LSST_AFW_GEOM_ELLIPSES_DISTORTION_H
#define LSST_AFW_GEOM_ELLIPSES_DISTORTION_H

/**
 *  \file
 *  \brief Definitions for Distortion and DistortionEllipse.
 *
 *  \note Do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/ellipses/EllipseImpl.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

class DistortionEllipse;

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
class Distortion : public detail::CoreImpl<Distortion,DistortionEllipse> {
    typedef detail::CoreImpl<Distortion,DistortionEllipse> Super;
public:

    enum Parameters { E1=0, E2=1, R=2 }; ///< Definitions for elements of a core vector.

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
    explicit Distortion(ParameterVector const & vector) : Super(vector) {}

    /// \brief Construct from parameter values.
    explicit Distortion(double e1=0, double e2=0, double radius=0) : Super(e1,e2,radius) {}
    
    /// \brief Construct from complex ellipticity and radius.
    explicit Distortion(std::complex<double> const & e, double radius=0) 
        : Super(e.real(),e.imag(),radius) {}

    /// \brief Converting copy constructor.
    Distortion(BaseCore const & other) { *this = other; }

    /// \brief Copy constructor.
    Distortion(Distortion const & other) : Super(other.getVector()) {}

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
 *  \brief An ellipse with a Distortion core.
 *
 *  \ingroup EllipseGroup
 */
class DistortionEllipse : public detail::EllipseImpl<Distortion,DistortionEllipse> {
    typedef detail::EllipseImpl<Distortion,DistortionEllipse> Super;
public:

    /// Definitions for elements of an ellipse vector.
    enum Parameters { X=0, Y=1, E1=2, E2=3, R=4 };

    /**
     *  \brief Set the parameters of this ellipse from another.
     *
     *  This does not change the parametrization of the ellipse.
     */
    DistortionEllipse & operator=(BaseEllipse const & other) { return Super::operator=(other); }

    /// \brief Set the parameters of this ellipse from another.
    DistortionEllipse & operator=(DistortionEllipse const & other) { return Super::operator=(other); }

    /// \brief Construct from a PointD and zero-size Core.
    explicit DistortionEllipse(PointD const & center = PointD()) : Super(new Distortion(),center) {}

    /// \brief Construct from a copy of a Distortion core.
    explicit DistortionEllipse(Distortion const & core, PointD const & center = PointD()) : 
        Super(core,center) {}

    /// \brief Construct from a 5-element parameter vector.
    explicit DistortionEllipse(BaseEllipse::ParameterVector const & vector) : Super(vector) {}

    /// \brief Converting copy constructor.
    DistortionEllipse(BaseEllipse const & other) : Super(other.getCore(),other.getCenter()) {}

    /// \brief Copy constructor.
    DistortionEllipse(DistortionEllipse const & other) : Super(other.getCore(),other.getCenter()) {}

};

} // namespace lsst::afw::geom::ellipses
}}} // namespace lsst::afw::geom
#endif // !LSST_AFW_GEOM_ELLIPSES_DISTORTION_H
