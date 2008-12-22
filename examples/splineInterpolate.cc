// -*- lsst-c++ -*-
#include <iostream>
#include <cmath>
#include <vector>

#include "boost/shared_ptr.hpp"
#include "lsst/afw/math/Interpolate.h"

using namespace std;
namespace math = lsst::afw::math;

namespace {
    double const NaN = std::numeric_limits<double>::quiet_NaN();
}

typedef math::LinearInterpolate<double,double> LinearT;
typedef math::SplineInterpolate<double,double> SplineT;

int main() {

    // create x,y vector<>s containing a sin() function
    int const nx = 20;
    vector<double> x(nx);
    vector<double> y(nx);
    double xlo = 0;
    double xhi = 2.0 * 3.14159;
    double range = xhi - xlo;
    
    for (int i = 0; i < nx; ++i) {
        x[i] = xlo + static_cast<double>(i)/(nx - 1) * range;
        y[i] = sin(x[i]);
    }


    // create a new x vector<> on a different grid and extending beyond the bounds
    //   of the interpolation range to tests extrapolation properties
    int const nx2 = 100;
    vector<double> x2(nx2);
    
    for (int i = 0; i < nx2; ++i) {
        x2[i] = xlo + ( ((nx + 2.0)/nx)*static_cast<double>(i)/(nx2 - 1) - 1.0/nx) * range;
    }
    
    // declare an spline interpolate object.  the constructor computes the derivatives
    SplineT yinterpS(x, y);

    // declare a linear interpolate object. the constructor computes the second derivatives
    LinearT yinterpL(x, y);
    
    // output the interpolated y values, 1st derivatives, and 2nd derivatives.
    for (int i = 0; i < nx2; ++i) {
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
