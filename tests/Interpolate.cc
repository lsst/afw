#include <iostream>
#include <cmath>
#include <vector>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Interpolate

#include "boost/test/unit_test.hpp"

#include "lsst/afw/math/Interpolate.h"

using namespace std;
namespace math = lsst::afw::math;

BOOST_AUTO_TEST_CASE(LinearInterpolateRamp) {

    int n = 10;
    vector<float> x(n);
    vector<float> y(n);
    for(int i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i);
    }
    float xtest = 4.5;

    {
        // === test the Linear interpolator ============================
        //math::InterpControl ictrl1(math::LINEAR, NaN, NaN);
        math::LinearInterpolate<float,float> yinterpL(x, y);
        float youtL = yinterpL.interpolate(xtest);

        BOOST_CHECK_EQUAL(youtL, xtest);
    }
}

BOOST_AUTO_TEST_CASE(SplineInterpolateRamp) {

    int n = 10;
    vector<float> x(n);
    vector<float> y(n);
    //float const NaN = std::numeric_limits<float>::quiet_NaN();
    for(int i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i);
    }
    float xtest = 4.5;

    {
        // === test the Spline interpolator =======================
        //math::InterpControl ictrl2(math::NATURAL_SPLINE, NaN, NaN);
        math::SplineInterpolate<float,float> yinterpS(x, y);
        float youtS = yinterpS.interpolate(xtest);
        
        BOOST_CHECK_EQUAL(youtS, xtest);
    }
}


BOOST_AUTO_TEST_CASE(SplineInterpolateParabola) {

    int const N = 20;
    vector<double> x(N);
    vector<double> y(N);
    double dydx = 1.0;
    double d2ydx2 = 0.5;
    double y0 = 10.0;
    
    //float const NaN = std::numeric_limits<float>::quiet_NaN();
    for(int i = 0; i < N; ++i) {
        x[i] = static_cast<double>(i);
        y[i] = d2ydx2*x[i]*x[i] + dydx*x[i] + y0;
    }
    double xtest = 9.5;
    double ytest = d2ydx2*xtest*xtest + dydx*xtest + y0;
    
    {
        // === test the Spline interpolator =======================
        math::InterpControl ictrl(math::NATURAL_SPLINE);
        ictrl.setDydx0( 2.0*d2ydx2*x[0] + dydx);
        ictrl.setDydxN( 2.0*d2ydx2*x[N - 1] + dydx);
        math::SplineInterpolate<double,double> yinterpS(x, y, ictrl);
        double youtS = yinterpS.interpolate(xtest);

        
        BOOST_CHECK_EQUAL(youtS, ytest);
    }
}


