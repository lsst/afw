// -*- LSST-C++ -*-
#include <iostream>
#include <cmath>
#include <vector>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Interpolate

#include "boost/test/unit_test.hpp"
#include "lsst/afw/math/Interpolate.h"

using namespace std;
namespace math = lsst::afw::math;

typedef math::Interpolate Interp;

BOOST_AUTO_TEST_CASE(LinearInterpolateRamp) {

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
        //math::InterpControl ictrl1(math::LINEAR, NaN, NaN);
        Interp yinterpL(x, y, ::gsl_interp_linear);
        double youtL = yinterpL.interpolate(xtest);

        BOOST_CHECK_EQUAL(youtL, xtest);
    }
}

BOOST_AUTO_TEST_CASE(SplineInterpolateRamp) {

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
        Interp yinterpS(x, y, ::gsl_interp_cspline);
        double youtS = yinterpS.interpolate(xtest);
        
        BOOST_CHECK_EQUAL(youtS, xtest);
    }
}


BOOST_AUTO_TEST_CASE(SplineInterpolateParabola) {

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
        Interp yinterpS(x, y, ::gsl_interp_akima);
        double youtS = yinterpS.interpolate(xtest);
        
        BOOST_CHECK_CLOSE(youtS, ytest, 1.0e-8);
    }
}


