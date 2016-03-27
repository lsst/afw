// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#ifndef LSST_AFW_GEOM_ELLIPSES_ConformalShear_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_ConformalShear_h_INCLUDED

#include "lsst/afw/geom/ellipses/EllipticityBase.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

class Distortion;

class ReducedShear;

/**
 *  @brief A logarithmic complex ellipticity with magnitude @f$|e| = \ln (a/b) @f$.
 *
 *  For a more complete definition, see Bernstein and Jarvis (2002); this the same as their
 *  conformal shear @f$\eta@f$ (eq. 2.3-2.6).
 */
class ConformalShear : public detail::EllipticityBase {
public:

    explicit ConformalShear(std::complex<double> const & complex) : detail::EllipticityBase(complex) {}

    explicit ConformalShear(double e1=0.0, double e2=0.0) : detail::EllipticityBase(e1, e2) {}

    ConformalShear(ConformalShear const & other) : detail::EllipticityBase(other.getComplex()) {}

    explicit ConformalShear(Distortion const & other) { this->operator=(other); }

    explicit ConformalShear(ReducedShear const & other) { this->operator=(other); }

    ConformalShear & operator=(ConformalShear const & other) {
        _complex = other._complex;
        return *this;
    }

    ConformalShear & operator=(Distortion const & other);

    ConformalShear & operator=(ReducedShear const & other);

    Jacobian dAssign(ConformalShear const & other) {
        _complex = other._complex;
        return Jacobian::Identity();
    }

    Jacobian dAssign(Distortion const & other);

    Jacobian dAssign(ReducedShear const & other);

    double getAxisRatio() const;

    void normalize() {}

    static std::string getName() { return "ConformalShear"; }

};

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_ConformalShear_h_INCLUDED
