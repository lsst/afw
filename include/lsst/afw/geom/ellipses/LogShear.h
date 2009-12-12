// -*- lsst-c++ -*-
#ifndef LSST_AFW_GEOM_ELLIPSES_LOGSHEAR_H
#define LSST_AFW_GEOM_ELLIPSES_LOGSHEAR_H

/**
 *  \file
 *  \brief Definitions for LogShear and LogShearEllipse.
 *
 *  \note Do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/ellipses/EllipseImpl.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

class LogShearEllipse;

/**
 *  \brief An ellipse core with logarithmic ellipticity and radius parameters.
 *
 *  The logarithmic shear \f$\gamma\f$ and convergence \f$\kappa\f$ are defined as
 *  \f[
 *    \gamma = \frac{1}{2}\ln\left(\frac{a}{b}\right)\,\exp\left(2i\theta\right); 
 *             \quad \quad \kappa = \frac{1}{2}\ln(ab)
 *  \f]
 *  where \f$(a,b,\theta)\f$ are the semimajor and semiminor axes and the position angle in radians.
 *  \f$\gamma\f$ thus behaves like a complex ellipticity with range \f$(-\infty,\infty)\f$ instead of
 *  \f$(-1,1)\f$, while \f$\kappa\f$ is exactly the logarithm of the geometric mean radius.
 *
 *  \ingroup EllipseGroup
 */
class LogShear : public detail::CoreImpl<LogShear,LogShearEllipse> {
    typedef detail::CoreImpl<LogShear,LogShearEllipse> Super;
public:
        
    ///\brief Definitions for elements of a core vector.
    enum Parameters { GAMMA1=0, GAMMA2=1, KAPPA=2 };

    /// Return a string that identifies this parametrization.
    virtual char const * getName() const { return "LogShear"; }

    /**
     *  \brief Put the parameters into a "standard form", if possible, and return
     *         false if they are entirely invalid.
     */
    virtual bool normalize() { return true; }

    /// \brief Increase the major and minor radii of the ellipse core by the given buffer.
    virtual void scale(double ratio) { _vector[KAPPA] += std::log(ratio); }

    /// \brief Return the logarithmic shear as a complex number.
    std::complex<double> getComplex() const { 
        return std::complex<double>(_vector[GAMMA1],_vector[GAMMA2]); 
    }

    /// \brief Set the logarithmic shear from a complex number.
    void setComplex(std::complex<double> const & gamma) { 
        _vector[GAMMA1] = gamma.real(); _vector[GAMMA2] = gamma.imag(); 
    }

    /// \brief Return the logarithmic shear magnitude.
    double getGamma() const {
        return std::sqrt(_vector[GAMMA1]*_vector[GAMMA1] + _vector[GAMMA2]*_vector[GAMMA2]); 
    }

    /// \brief Set the logarithmic shear magnitude, keeping the position angle constant.
    void setGamma(double gamma) { 
        double f = gamma/getGamma(); _vector[GAMMA1] *= f; _vector[GAMMA2] *= f; 
    }

    /// \brief Standard assignment.
    LogShear & operator=(LogShear const & other) { _vector = other._vector; return *this; }

    /// \brief Converting assignment.
    virtual LogShear & operator=(BaseCore const & other) { other._assignTo(*this); return *this; }

    /// \brief Construct from a parameter vector.
    explicit LogShear(ParameterVector const & vector) : Super(vector) {}

    /// \brief Construct from parameter values.
    explicit LogShear(double gamma1=0, double gamma2=0,
                      double kappa=-std::numeric_limits<double>::infinity()) : 
        Super(gamma1,gamma2,kappa) {}

    /// \brief Construct from complex logarithmic shear and convergence.
    explicit LogShear(std::complex<double> const & gamma,
                      double kappa=-std::numeric_limits<double>::infinity()) : 
        Super(gamma.real(),gamma.imag(),kappa) {}

    /// \brief Converting copy constructor.
    LogShear(BaseCore const & other) { *this = other; }

    /// \brief Copy constructor.
    LogShear(LogShear const & other) : Super(other.getVector()) {}

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
 *  \brief An ellipse with a LogShear core.
 *
 *  \ingroup EllipseGroup
 */
class LogShearEllipse : public detail::EllipseImpl<LogShear,LogShearEllipse> {
    typedef detail::EllipseImpl<LogShear,LogShearEllipse> Super;
public:

    /// Definitions for elements of an ellipse vector.
    enum Parameters { X=0, Y=1, GAMMA1=2, GAMMA2=3, KAPPA=4 };

    /**
     *  \brief Set the parameters of this ellipse from another.
     *
     *  This does not change the parametrization of the ellipse.
     */
    LogShearEllipse & operator=(BaseEllipse const & other) { return Super::operator=(other); }

    /// \brief Set the parameters of this ellipse from another.
    LogShearEllipse & operator=(LogShearEllipse const & other) { return Super::operator=(other); }

    /// \brief Construct from a PointD and zero-size Core.
    explicit LogShearEllipse(PointD const & center = PointD()) : Super(new LogShear(),center) {}

    /// \brief Construct from a copy of a LogShear core.
    explicit LogShearEllipse(LogShear const & core, PointD const & center = PointD()) : 
        Super(core,center) {}

    /// \brief Construct from a 5-element parameter vector.
    explicit LogShearEllipse(BaseEllipse::ParameterVector const & vector) : Super(vector) {}

    /// \brief Converting copy constructor.
    LogShearEllipse(BaseEllipse const & other) : Super(other.getCore(),other.getCenter()) {}

    /// \brief Copy constructor.
    LogShearEllipse(LogShearEllipse const & other) : Super(other.getCore(),other.getCenter()) {}

};

} // namespace lsst::afw::geom::ellipses
}}} // namespace lsst::afw::geom
#endif // !LSST_AFW_GEOM_ELLIPSES_LOGSHEAR_H
