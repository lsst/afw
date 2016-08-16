// -*- LSST-C++ -*-

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
 * I've chosen a parabola here: f(x) = k + kx*x*x
 * as it's got an easy-to-check analytic answer.
 *
 * @note We *have* to inherit from unary_function<>
 */
template<typename IntegrandT>
class Parab1D : public std::unary_function<IntegrandT, IntegrandT>  {
public:

    // declare coefficients at instantiation
    Parab1D(double const k, double const kx) : _k(k), _kx(kx) {}

    // operator() must be overloaded to return the evaluation of the function
    // ** This is the function to be integrated **
    IntegrandT operator()(IntegrandT const x) const {
        return (_k - _kx*x*x);
    }

    // for this example we have an analytic answer to check
    IntegrandT getAnalyticArea(IntegrandT const x1, IntegrandT const x2) {
        return _k*(x2 - x1) - _kx*(x2*x2*x2 - x1*x1*x1)/3.0;
    }

private:
    double _k, _kx;
};



/** =========================================================================
 * define a simple 2D function as a functor to be integrated.
 * I've chosen a 2D paraboloid: f(x) = k - kx*x*x - ky*y*y
 * as it's got an easy-to-check analytic answer.
 *
 * @note we *have* to inherit from binary_function<>
 */
template<typename IntegrandT>
class Parab2D : public std::binary_function<IntegrandT, IntegrandT, IntegrandT> {
public:
    // declare coefficients at instantiation.
    Parab2D(IntegrandT const k, IntegrandT const kx, IntegrandT const ky) : _k(k), _kx(kx), _ky(ky) {}

    // operator() must be overloaded to return the evaluation of the function
    // ** This is the function to be integrated **
    // @note *need*  ... operator()() __const__ {
    IntegrandT operator()(IntegrandT const x, IntegrandT const y) const {
        return (_k - _kx*x*x - _ky*y*y);
    }

    // for this example we have an analytic answer to check
    IntegrandT getAnalyticVolume(IntegrandT const x1, IntegrandT const x2,
                                 IntegrandT const y1, IntegrandT const y2) {
        IntegrandT const xw = x2 - x1;
        IntegrandT const yw = y2 - y1;
        return _k*xw*yw - _kx*(x2*x2*x2 - x1*x1*x1)*yw/3.0 - _ky*(y2*y2*y2 - y1*y1*y1)*xw/3.0;
    }

private:
    IntegrandT _k, _kx, _ky;
};



/** =============================================================================
 * Define a pair of normal functions that do the same thing as the above functors
 *
 */

// the 1D parabola
double parabola(double const x) {
    double const k = 100.0, kx = 1.0;
    return k - kx*x*x;
}
// the 2D paraboloid
double parabola2d(double const x, double const y) {
    double const k = 100.0, kx = 1.0, ky = 1.0;
    return k - kx*x*x - ky*y*y;
}



/** =====================================================================
 *  Main body of code
 *  ======================================================================
 */
int main() {

    // set limits of integration
    double const x1 = 0, x2 = 9, y1 = 0, y2 = 9;
    // set the coefficients for the quadratic equation
    // (parabola f(x) = k + kx*x*x + ky*y*y)
    double const k = 100, kx = 1.0, ky = 1.0;


    // ==========   1D integrator ==========

    // instantiate a Parab1D Functor
    Parab1D<double> parab1d(k, kx);

    // integrate the area under the curve, and then get the analytic result
    double const parab_area_integrate  = math::integrate(parab1d, x1, x2);
    double const parab_area_analytic = parab1d.getAnalyticArea(x1, x2);

    // now run it on the 1d function (you *need* to wrap the function in ptr_fun())
    double const parab_area_integrate_func = math::integrate(std::ptr_fun(parabola), x1, x2);

    // output
    std::cout << "1D integrate: functor = " << parab_area_integrate <<
        "  function = " << parab_area_integrate_func <<
        "  analytic = " << parab_area_analytic << std::endl;


    // ==========  2D integrator ==========

    // instantiate a Parab2D
    Parab2D<double> parab2d(k, kx, ky);

    // integrate the volume under the function, and then get the analytic result
    double const parab_volume_integrate  = math::integrate2d(parab2d, x1, x2, y1, y2);
    double const parab_volume_analytic = parab2d.getAnalyticVolume(x1, x2, y1, y2);

    // now run it on the 2d function (you *need* to wrap the function in ptr_fun())
    double const parab_volume_integrate_func = math::integrate2d(std::ptr_fun(parabola2d), x1, x2, y1, y2);

    // output
    std::cout << "2D integrate: functor = " << parab_volume_integrate <<
        "  function = " << parab_volume_integrate_func <<
        "  analytic = " << parab_volume_analytic << std::endl;

    return 0;
}
