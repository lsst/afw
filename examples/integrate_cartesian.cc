// -*- lsst-c++ -*-
/**
 * @file   rombergIntegrate.cc
 * @author S. Bickerton
 * @date   May 25, 2009
 *
 * This example demonstrates how to use the romberg 1D and 2D
 * integrators in afw::math (Quadrature).
 *
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>

#include "lsst/afw/math/Integrate.h"

namespace math = lsst::afw::math;

/** =========================================================================
 * define a simple 1D function as a functor to be integrated.
 * I've chosen a parabola here: f(x) = K + kx*x*x
 * as it's got an easy-to-check analytic answer.
 * 
 * @note We *have* to inherit from unary_function<>
 */
template<typename IntegrandT>
class Parab1D : public std::unary_function<IntegrandT,IntegrandT>  {
public:
    
    // declare coefficients at instantiation
    Parab1D(double const K, double const kx) : _K(K), _kx(kx) {}
    
    // operator() must be overloaded to return the evaluation of the function
    // ** This is the function to be integrated **
    IntegrandT operator()(IntegrandT const x) const {
        return (_K - _kx*x*x);
    }

    // for this example we have an analytic answer to check
    IntegrandT getAnalyticArea(IntegrandT const x1, IntegrandT const x2) {
        return _K*(x2-x1) - _kx*(x2*x2*x2-x1*x1*x1)/3.0;
    }
    
private:
    double _K, _kx;
};



/** =========================================================================
 * define a simple 2D function as a functor to be integrated.
 * I've chosen a 2D paraboloid: f(x) = K - kx*x*x - ky*y*y
 * as it's got an easy-to-check analytic answer.
 *
 * @note we *have* to inherit from binary_function<>
 */
template<typename IntegrandT>
class Parab2D : public std::binary_function<IntegrandT,IntegrandT,IntegrandT> {
public:
    // declare coefficients at instantiation.
    Parab2D(IntegrandT const K, IntegrandT const kx, IntegrandT const ky) : _K(K), _kx(kx), _ky(ky) {}
    
    // operator() must be overloaded to return the evaluation of the function
    // ** This is the function to be integrated **
    // @note *need*  ... operator()() __const__ {
    IntegrandT operator()(IntegrandT const x, IntegrandT const y) const {
        return (_K - _kx*x*x - _ky*y*y);
    }

    // for this example we have an analytic answer to check
    IntegrandT getAnalyticVolume(IntegrandT const x1, IntegrandT const x2,
                                 IntegrandT const y1, IntegrandT const y2) {
        IntegrandT const xw = x2 - x1;
        IntegrandT const yw = y2 - y1;
        return _K*xw*yw - _kx*(x2*x2*x2-x1*x1*x1)*yw/3.0 - _ky*(y2*y2*y2-y1*y1*y1)*xw/3.0;
    }

private:
    IntegrandT _K, _kx, _ky;
};



/** =============================================================================
 * Define a pair of normal functions that do the same thing as the above functors
 *
 */

// the 1D parabola
double parabola(double const x) {
    double const K = 100.0, kx = 1.0;
    return K - kx*x*x;
}
// the 2D paraboloid
double parabola2d(double const x, double const y) {
    double const K = 100.0, kx = 1.0, ky = 1.0;
    return K - kx*x*x - ky*y*y;
}



/** =====================================================================
 *  Main body of code
 *  ======================================================================
 */
int main() {

    // set limits of integration
    double const X1 = 0, X2 = 9, Y1 = 0, Y2 = 9;
    // set the coefficients for the quadratic equation
    // (parabola f(x) = K + KX*x*x + KY*y*y)
    double const K = 100, KX = 1.0, KY = 1.0;

    
    // ==========   1D integrator ==========

    // instantiate a Parab1D Functor
    Parab1D<double> parab1d(K, KX);
    
    // integrate the area under the curve, and then get the analytic result
    double const parab_area_integrate  = math::integrate(parab1d, X1, X2);
    double const parab_area_analytic = parab1d.getAnalyticArea(X1, X2);

    // now run it on the 1d function (you *need* to wrap the function in ptr_fun())
    double const parab_area_integrate_func = math::integrate(std::ptr_fun(parabola), X1, X2);

    // output
    std::cout << "1D integrate: functor = " << parab_area_integrate << 
        "  function = " << parab_area_integrate_func <<
        "  analytic = " << parab_area_analytic << std::endl;

    
    // ==========  2D integrator ==========

    // instantiate a Parab2D
    Parab2D<double> parab2d(K, KX, KY);

    // integrate the volume under the function, and then get the analytic result
    double const parab_volume_integrate  = math::integrate2d(parab2d, X1, X2, Y1, Y2);
    double const parab_volume_analytic = parab2d.getAnalyticVolume(X1, X2, Y1, Y2);

    // now run it on the 2d function (you *need* to wrap the function in ptr_fun())
    double const parab_volume_integrate_func = math::integrate2d(std::ptr_fun(parabola2d), X1, X2, Y1, Y2);

    // output
    std::cout << "2D integrate: functor = " << parab_volume_integrate <<
        "  function = " << parab_volume_integrate_func <<
        "  analytic = " << parab_volume_analytic << std::endl;
    
    return 0;
}
