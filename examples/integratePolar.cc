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

/*
 * This example demonstrates how to use the romberg2D()
 * integrator (afw::math Quadrature) with a polar function.
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>

#include "lsst/afw/math/Integrate.h"

namespace math = lsst::afw::math;

/* =========================================================================
 * define a simple 2D function as a functor to be integrated.
 * I've chosen a paraboloid: f(x) = K - kr*r*r
 * as it's got an easy-to-check analytic answer.
 *
 * @note that we *have* to inherit from binary_function<>
 */
template<typename IntegrandT>
class Parab2D : public std::binary_function<IntegrandT, IntegrandT, IntegrandT> {
public:
    // declare coefficients at instantiation.
    Parab2D(IntegrandT const k, IntegrandT const kr) : _k(k), _kr(kr) {}

    // operator() must be overloaded to return the evaluation of the function
    // ** This is the function to be integrated **
    //
    // NOTE: extra 'r' term due to polar coords (ie. the 'r' in r*dr*dtheta)
    IntegrandT operator()(IntegrandT const r, IntegrandT const) const {
        return (_k - _kr*r*r)*r;
    }

    // for this example we have an analytic answer to check
    IntegrandT getAnalyticVolume(IntegrandT const r1, IntegrandT const r2,
                                 IntegrandT const theta1, IntegrandT const theta2) {
        return ((theta2 - theta1) *
                ((0.5*_k*r2*r2 - (1.0/3.0)*_kr*r2*r2*r2) -
                 (0.5*_k*r1*r1 - (1.0/3.0)*_kr*r1*r1*r1) ));
    }

private:
    IntegrandT _k, _kr;
};



/* =============================================================================
 * Define a normal function that does the same thing as the above functor
 *
 */
double parabola2d(double const r, double const) {
    double const k = 1.0, kr = 0.0;
    return (k - kr*r*r)*r;
}



/* =====================================================================
 *  Main body of code
 *  ======================================================================
 */
int main() {

    // set limits of integration
    double const r1 = 0, r2 = 1, theta1 = 0, theta2 = 2.0*M_PI;
    // set the coefficients for the quadratic equation
    // (parabola f(r) = k - kr*r*r)
    double const k = 1.0, kr = 0.0;  // not really a parabola ... force the answer to be 'pi'


    // ==========  2D integrator ==========

    // instantiate a Parab2D
    Parab2D<double> parab2d(k, kr);

    // integrate the volume under the function, and then get the analytic result
    double const parab_volume_integrate  = math::integrate2d(parab2d, r1, r2, theta1, theta2);
    double const parab_volume_analytic = parab2d.getAnalyticVolume(r1, r2, theta1, theta2);

    // now run it on the 2d function (you *need* to wrap the function in ptr_fun())
    double const parab_volume_integrate_func =
        math::integrate2d(std::ptr_fun(parabola2d), r1, r2, theta1, theta2);

    // output
    std::cout << "2D integrate: functor = " << parab_volume_integrate <<
        "  function = " << parab_volume_integrate_func <<
        "  analytic = " << parab_volume_analytic << std::endl;

    return 0;
}
