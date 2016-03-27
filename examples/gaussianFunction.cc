/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>

#include "boost/format.hpp"

#include "lsst/afw/math/FunctionLibrary.h"

using namespace std;

int main() {
    typedef double FuncReturn;
    double sigma = 2.0;
    lsst::afw::math::GaussianFunction1<FuncReturn> gaussFunc(sigma);

    for (double x = -3.0; x < 3.1; x += 0.5) {
        cout << boost::format("gauss(%5.1f) = %6.3f\n") % x % gaussFunc(x);
    }
}
