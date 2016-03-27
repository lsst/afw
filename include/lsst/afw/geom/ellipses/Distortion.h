// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#ifndef LSST_AFW_GEOM_ELLIPSES_Distortion_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_Distortion_h_INCLUDED

#include "lsst/afw/geom/ellipses/EllipticityBase.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

class ConformalShear;
class ReducedShear;

/**
 *  @brief A complex ellipticity with magnitude \f$|e| = \frac{a^2 - b^2}{a^2 + b^2}\f$.
 *
 *  For a more complete definition, see Bernstein and Jarvis (2002); this the same as their
 *  distortion @f$\delta@f$ (eq. 2.7).
 */
class Distortion : public detail::EllipticityBase {
public:

    explicit Distortion(std::complex<double> const & complex) : detail::EllipticityBase(complex) {}

    explicit Distortion(double e1=0.0, double e2=0.0) : detail::EllipticityBase(e1, e2) {}

    Distortion(Distortion const & other) : detail::EllipticityBase(other.getComplex()) {}

    explicit Distortion(ConformalShear const & other) { this->operator=(other); }

    explicit Distortion(ReducedShear const & other) { this->operator=(other); }

    Distortion & operator=(Distortion const & other) {
        _complex = other._complex;
        return *this;
    }

    Distortion & operator=(ConformalShear const & other);

    Distortion & operator=(ReducedShear const & other);

    Jacobian dAssign(Distortion const & other) {
        _complex = other._complex;
        return Jacobian::Identity();
    }

    Jacobian dAssign(ConformalShear const & other);

    Jacobian dAssign(ReducedShear const & other);

    double getAxisRatio() const;

    void normalize();

    static std::string getName() { return "Distortion"; }

};

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_Distortion_h_INCLUDED
