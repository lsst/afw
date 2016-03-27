// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
/**
 * @file   Integrate.cc
 * @author S. Bickerton
 * @date   May 25, 2009
 *
 * This test evaluates the 1D and 2D integrators
 * integrators in the afw::math (Integrate) suite.
 *
 * Outline:
 *
 * Integrate functions which have analytic solutions and compare.
 * I chose parabolae.
 *
 */
#include <cmath>
#include <vector>
#include <functional>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Integrate

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop

#include "lsst/afw/math/Integrate.h"

using namespace std;
namespace math = lsst::afw::math;


/* define a simple 1D function as a functor to be integrated.
 * I've chosen a parabola here: f(x) = k + kx*x*x
 * as it's got an easy-to-check analytic answer.
 *
 * We have to inherit from IntegrandBase for the integrator to work.
 */
template<typename IntegrandT>
class Parab1D : public std::unary_function<IntegrandT, IntegrandT> {
public:
    
    Parab1D(double k, double kx) : _k(k), _kx(kx) {}
    
    // for this example we have an analytic answer to check
    double getAnalyticArea(double const x1, double const x2) {
        return _k*(x2 - x1) - _kx*(x2*x2*x2 - x1*x1*x1)/3.0;
    }
    
    // operator() must be overloaded to return the evaluation of the function
    IntegrandT operator() (IntegrandT const x) const { return (_k - _kx*x*x); }
    
private:
    double _k, _kx;
};

double parabola1d(double x) {
    double k = 100.0, kx = 1.0;
    return k - kx*x*x;
}

/* define a simple 2D function as a functor to be integrated.
 * I've chosen a 2D paraboloid: f(x) = k - kx*x*x - ky*y*y
 * as it's got an easy-to-check analytic answer.
 *
 * Note that we have to inherit from IntegrandBase
 */
template<typename IntegrandT>
class Parab2D : public std::binary_function<IntegrandT, IntegrandT, IntegrandT> {
public:
    Parab2D(double k, double kx, double ky) : _k(k), _kx(kx), _ky(ky) {}
    
    // for this example we have an analytic answer to check
    double getAnalyticVolume(double const x1, double const x2, double const y1, double const y2) {
        double const xw = x2 - x1;
        double const yw = y2 - y1;
        return _k*xw*yw - _kx*(x2*x2*x2 - x1*x1*x1)*yw/3.0 - _ky*(y2*y2*y2 - y1*y1*y1)*xw/3.0;
    }
    
    // operator() must be overloaded to return the evaluation of the function
    IntegrandT operator() (IntegrandT const x, IntegrandT const y) const {
        return (_k - _kx*x*x - _ky*y*y);
    }
    
private:
    double _k, _kx, _ky;
};


double parabola2d(double const x, double const y) {
    double const k = 100.0, kX = 1.0, kY = 1.0;
    return k - kX*x*x - kY*y*y;
}

/**
 * @brief Test the 1D integrator on a Parabola
 * @note default precision is 1e-6 for integrate()
 */
BOOST_AUTO_TEST_CASE(Parabola1D) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */

    // set limits of integration
    double x1 = 0, x2 = 9;
    // set the coefficients for the quadratic equation (parabola f(x) = k + kx*x*x)
    double k = 100, kx = 1.0;

    // ==========   The 1D integrator ==========
    // instantiate a Parab1D Functor, integrate numerically, and analytically
    Parab1D<double> parab1d(k, kx);
    double parab_area_integrate  = math::integrate(parab1d, x1, x2);
    double parab_area_analytic = parab1d.getAnalyticArea(x1, x2);

    double parab_area_integrate_function = math::integrate(std::ptr_fun(parabola1d), x1, x2);
    
    BOOST_CHECK_CLOSE(parab_area_integrate, parab_area_analytic, 1e-6);
    BOOST_CHECK_CLOSE(parab_area_integrate_function, parab_area_analytic, 1e-6);
}


/**
 * @brief Test the 2d integrator on a Paraboloid
 * @note default precision is 1e-6 from integrate2d()
 */
BOOST_AUTO_TEST_CASE(Parabola2D) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */

    // set limits of integration
    double x1 = 0, x2 = 9, y1 = 0, y2 = 9;
    // set the coefficients for the quadratic equation (parabola f(x) = k + kx*x*x + ky*y*y)
    double k = 100, kx = 1.0, ky = 1.0;

    // ==========   The 2D integrator ==========
    // instantiate a Parab2D, integrate numerically and analytically
    Parab2D<double> parab2d(k, kx, ky);
    double parab_volume_integrate  = math::integrate2d(parab2d, x1, x2, y1, y2);
    double parab_volume_analytic = parab2d.getAnalyticVolume(x1, x2, y1, y2);

    double parab_volume_integrate_function = math::integrate2d(std::ptr_fun(parabola2d), x1, x2, y1, y2);

    BOOST_CHECK_CLOSE(parab_volume_integrate, parab_volume_analytic, 1e-6);
    BOOST_CHECK_CLOSE(parab_volume_integrate_function, parab_volume_analytic, 1e-6);
}

