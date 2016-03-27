// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>
#include <cmath>
#include <vector>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Interpolate

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "lsst/afw/math/Interpolate.h"

using namespace std;
namespace math = lsst::afw::math;

BOOST_AUTO_TEST_CASE(LinearInterpolateRamp) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */

    int n = 10;
    vector<double> x(n);
    vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i);
        y[i] = static_cast<double>(i);
    }
    double xtest = 4.5;

    {
        // === test the Linear interpolator ============================
        //math::InterpControl ictrl1(math::Interpolate::LINEAR, NaN, NaN);
        PTR(math::Interpolate) yinterpL = math::makeInterpolate(x, y, math::Interpolate::LINEAR);
        double youtL = yinterpL->interpolate(xtest);

        BOOST_CHECK_EQUAL(youtL, xtest);
    }
}

BOOST_AUTO_TEST_CASE(SplineInterpolateRamp) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */

    int n = 10;
    vector<double> x(n);
    vector<double> y(n);
    //double const NaN = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i);
        y[i] = static_cast<double>(i);
    }
    double xtest = 4.5;

    {
        // === test the Spline interpolator =======================
        //math::InterpControl ictrl2(math::NATURAL_SPLINE, NaN, NaN);
        PTR(math::Interpolate) yinterpS = math::makeInterpolate(x, y, math::Interpolate::CUBIC_SPLINE);
        double youtS = yinterpS->interpolate(xtest);
        
        BOOST_CHECK_EQUAL(youtS, xtest);
    }
}


BOOST_AUTO_TEST_CASE(SplineInterpolateParabola) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */

    int const n = 20;
    vector<double> x(n);
    vector<double> y(n);
    double dydx = 1.0;
    double d2ydx2 = 0.5;
    double y0 = 10.0;
    
    //double const NaN = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i);
        y[i] = d2ydx2*x[i]*x[i] + dydx*x[i] + y0;
    }
    double xtest = 9.5;
    double ytest = d2ydx2*xtest*xtest + dydx*xtest + y0;
    
    {
        // === test the Spline interpolator =======================
        PTR(math::Interpolate) yinterpS = math::makeInterpolate(x, y, math::Interpolate::AKIMA_SPLINE);
        double youtS = yinterpS->interpolate(xtest);
        
        BOOST_CHECK_CLOSE(youtS, ytest, 1.0e-8);
    }
}


