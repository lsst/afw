// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#ifndef LSST_AFW_GEOM_ELLIPSES_Separable_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_Separable_h_INCLUDED

/**
 *  \file
 *  @brief Definitions and inlines for Separable.
 *
 *  \note Do not include directly; use the main ellipse header file.
 */

#include "lsst/afw/geom/ellipses/BaseCore.h"
#include "lsst/afw/geom/ellipses/Convolution.h"
#include "lsst/afw/geom/ellipses/Transformer.h"
#include "lsst/afw/geom/ellipses/GridTransform.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

/**
 *  @brief An ellipse core with a complex ellipticity and radius parameterization.
 *
 *  
 */
template <typename Ellipticity_, typename Radius_>
class Separable : public BaseCore {
public:

    typedef boost::shared_ptr<Separable> Ptr;
    typedef boost::shared_ptr<Separable const> ConstPtr;

    enum ParameterEnum { E1=0, E2=1, RADIUS=2 }; ///< Definitions for elements of a core vector.

    typedef Ellipticity_ Ellipticity;
    typedef Radius_ Radius;

    double const getE1() const { return _ellipticity.getE1(); }
    void setE1(double e1) { _ellipticity.setE1(e1); }

    double const getE2() const { return _ellipticity.getE2(); }
    void setE2(double e2) { _ellipticity.setE2(e2); }

    Radius const & getRadius() const { return _radius; }
    Radius & getRadius() { return _radius; }
    void setRadius(double radius) { _radius = radius; }
    void setRadius(Radius const & radius) { _radius = radius; }

    Ellipticity const & getEllipticity() const { return _ellipticity; }
    Ellipticity & getEllipticity() { return _ellipticity; }

    /// @brief Deep copy the ellipse core.
    Ptr clone() const { return boost::static_pointer_cast<Separable>(_clone()); }

    /// Return a string that identifies this parametrization.
    virtual std::string getName() const;

    /**
     *  @brief Put the parameters into a "standard form", and throw InvalidEllipseParameters
     *         if they cannot be normalized.
     */
    virtual void normalize();

    virtual void readParameters(double const * iter);

    virtual void writeParameters(double * iter) const;

    /// @brief Standard assignment.
    Separable & operator=(Separable const & other);

    /// @brief Converting assignment.
    Separable & operator=(BaseCore const & other) { BaseCore::operator=(other); return *this; }

    /// @brief Construct from parameter values.
    explicit Separable(double e1=0.0, double e2=0.0, double radius=Radius(), bool normalize=true);

    /// @brief Construct from parameter values.
    explicit Separable(std::complex<double> const & complex, 
                       double radius=Radius(), bool normalize=true);

    /// @brief Construct from parameter values.
    explicit Separable(Ellipticity const & ellipticity, double radius=Radius(), bool normalize=true);

    /// @brief Construct from a parameter vector.
    explicit Separable(BaseCore::ParameterVector const & vector, bool normalize=false);

    /// @brief Copy constructor.
    Separable(Separable const & other) : _ellipticity(other._ellipticity), _radius(other._radius) {}

    /// @brief Converting copy constructor.
    Separable(BaseCore const & other) { *this = other; }

#ifndef SWIG
    /// @brief Converting copy constructor.
    Separable(BaseCore::Transformer const & transformer) {
        transformer.apply(*this);
    }

    /// @brief Converting copy constructor.
    Separable(BaseCore::Convolution const & convolution) {
        convolution.apply(*this);
    }
#endif
protected:

    virtual BaseCore::Ptr _clone() const { return boost::make_shared<Separable>(*this); }

    virtual void _assignToQuadrupole(double & ixx, double & iyy, double & ixy) const;
    virtual void _assignFromQuadrupole(double ixx, double iyy, double ixy);

    virtual void _assignToAxes(double & a, double & b, double & theta) const;
    virtual void _assignFromAxes(double a, double b, double theta);

    virtual Jacobian _dAssignToQuadrupole(double & ixx, double & iyy, double & ixy) const;
    virtual Jacobian _dAssignFromQuadrupole(double ixx, double iyy, double ixy);

    virtual Jacobian _dAssignToAxes(double & a, double & b, double & theta) const;
    virtual Jacobian _dAssignFromAxes(double a, double b, double theta);

private:

    static BaseCore::Registrar<Separable> registrar;

    Ellipticity _ellipticity;
    Radius _radius;
};

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_Separable_h_INCLUDED
