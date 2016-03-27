// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#ifndef LSST_AFW_GEOM_ELLIPSES_EllipticityBase_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_EllipticityBase_h_INCLUDED

#include "Eigen/Core"
#include <complex>

namespace lsst { namespace afw { namespace geom { namespace ellipses {

namespace detail {

/**
 *  @brief EllipticityBase is a base class for complex ellipticity types.
 *
 *  EllipticityBase does not have a virtual destructor, and only exists
 *  for code reuse purposes.  The ellipticity classes are not polymorphic
 *  simply to keep them small.
 */
class EllipticityBase {
public:

    typedef Eigen::Matrix2d Jacobian;

    enum ParameterEnum { E1=0, E2=1 };

    std::complex<double> & getComplex() { return _complex; }

    std::complex<double> const & getComplex() const { return _complex; }

    void setComplex(std::complex<double> const & v) { _complex = v; }

    double getE1() const { return _complex.real(); }
    void setE1(double e1) {
#if __cplusplus < 201103L
        _complex = std::complex<double>(e1, _complex.imag());
#else
        _complex.real(e1);
#endif
    }

    double getE2() const { return _complex.imag(); }
    void setE2(double e2) {
#if __cplusplus < 201103L
        _complex = std::complex<double>(_complex.real(), e2);
#else
        _complex.imag(e2);
#endif
    }

    double getE() const { return std::sqrt(std::norm(_complex)); }
    void setE(double e) { _complex *= e / getE(); }

    double getTheta() const { return 0.5 * std::arg(_complex); }

protected:

    explicit EllipticityBase(std::complex<double> const & complex) : _complex(complex) {}

    explicit EllipticityBase(double e1=0.0, double e2=0.0) : _complex(e1, e2) {}

    std::complex<double> _complex;
};

} // namespace detail

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_EllipticityBase_h_INCLUDED
