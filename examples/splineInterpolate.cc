// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>
#include <cmath>
#include <vector>

#include "boost/shared_ptr.hpp"
#include "lsst/afw/math/Interpolate.h"

using namespace std;
namespace math = lsst::afw::math;

int main() {

    // create x,y vector<>s containing a sin() function
    int const nX = 20;
    vector<double> x(nX);
    vector<double> y(nX);
    double const xLo = 0;
    double const xHi = 2.0*M_PI;
    double const range = xHi - xLo;
    
    for (int i = 0; i < nX; ++i) {
        x[i] = xLo + static_cast<double>(i)/(nX - 1) * range;
        y[i] = sin(x[i]);
    }

    // create a new x vector<> on a different grid and extending beyond the bounds
    //   of the interpolation range to tests extrapolation properties
    int const nX2 = 100;
    vector<double> x2(nX2);
    for (int i = 0; i < nX2; ++i) {
        x2[i] = xLo + ( ((nX + 2.0)/nX)*static_cast<double>(i)/(nX2 - 1) - 1.0/nX) * range;
    }
    
    // declare an spline interpolate object.  the constructor computes the first derivatives
    PTR(math::Interpolate) yinterpS = math::makeInterpolate(x, y, math::Interpolate::LINEAR);
    
    // declare a linear interpolate object. the constructor computes the second derivatives
    PTR(math::Interpolate) yinterpL = math::makeInterpolate(x, y, math::Interpolate::CUBIC_SPLINE);
    
    // output the interpolated y values, 1st derivatives, and 2nd derivatives.
    for (int i = 0; i < nX2; ++i) {
        cout << i << " " << x2[i] << " " <<
            yinterpL->interpolate(x2[i]) << " " <<
            yinterpS->interpolate(x2[i]) << " " <<
            endl;
    }

    return 0;
}
