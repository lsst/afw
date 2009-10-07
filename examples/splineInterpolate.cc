// -*- lsst-c++ -*-
#include <iostream>
#include <cmath>
#include <vector>

#include "boost/shared_ptr.hpp"
#include "lsst/afw/math/Interpolate.h"

using namespace std;
namespace math = lsst::afw::math;

typedef math::LinearInterpolate<double,double> Linear;
typedef math::SplineInterpolate<double,double> Spline;

int main() {

    // create x,y vector<>s containing a sin() function
    int const NX = 20;
    vector<double> x(NX);
    vector<double> y(NX);
    double const XLO = 0;
    double const XHI = 2.0*M_PI;
    double const range = XHI - XLO;
    
    for (int i = 0; i < NX; ++i) {
        x[i] = XLO + static_cast<double>(i)/(NX - 1) * range;
        y[i] = sin(x[i]);
    }


    // create a new x vector<> on a different grid and extending beyond the bounds
    //   of the interpolation range to tests extrapolation properties
    int const NX2 = 100;
    vector<double> x2(NX2);
    for (int i = 0; i < NX2; ++i) {
        x2[i] = XLO + ( ((NX + 2.0)/NX)*static_cast<double>(i)/(NX2 - 1) - 1.0/NX) * range;
    }
    
    // declare an spline interpolate object.  the constructor computes the first derivatives
    Spline yinterpS(x, y);

    // declare a linear interpolate object. the constructor computes the second derivatives
    Linear yinterpL(x, y);
    
    // output the interpolated y values, 1st derivatives, and 2nd derivatives.
    for (int i = 0; i < NX2; ++i) {
        cout << i << " " << x2[i] << " " <<
            yinterpL.interpolate(x2[i]) << " " <<
            yinterpS.interpolate(x2[i]) << " " <<
            yinterpL.interpolateDyDx(x2[i]) << " " <<
            yinterpS.interpolateDyDx(x2[i]) << " " <<
            yinterpL.interpolateD2yDx2(x2[i]) << " " <<
            yinterpS.interpolateD2yDx2(x2[i]) << " " <<
            endl;
    }

    return 0;
}
