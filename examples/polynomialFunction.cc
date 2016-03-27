/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>
#include <vector>

#include "boost/format.hpp"

#include "lsst/afw/math/FunctionLibrary.h"

using namespace std;

int main() {
    typedef double FuncReturn;
    const unsigned int order = 2;
    vector<double> params(order+1);
    lsst::afw::math::PolynomialFunction1<FuncReturn> polyFunc(order);

    for (unsigned int ii = 0; ii < params.size(); ++ii) {
        params[ii] = static_cast<double>(1 + order - ii);
    }
    polyFunc.setParameters(params);
    
    cout << "Polynomial function with parameters: ";
    for (unsigned int ii = 0; ii < params.size(); ++ii) {
        cout << params[ii] << " ";
    }
    cout << endl << endl;

    for (double x = -3.0; x < 3.1; x += 0.5) {
        cout << boost::format("poly(%5.1f) = %6.3f\n") % x % polyFunc(x);
    }
}
