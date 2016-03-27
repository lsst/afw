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
    typedef double funcType;
    const unsigned int order = 3;
    vector<double> params(order+1);
    lsst::afw::math::Chebyshev1Function1<funcType> chebyFunc(order);

    for (unsigned int jj = 0; jj < params.size(); ++jj) {
        for (unsigned int ii = 0; ii < params.size(); ++ii) {
            params[ii] = (ii == jj) ? 1.0 : 0.0;
        }
        chebyFunc.setParameters(params);
        cout << "Chebychev polynomial of the first kind with parameters: ";
        for (unsigned int ii = 0; ii < params.size(); ++ii) {
            cout << params[ii] << " ";
        }
        cout << endl << endl;
        for (double x = -1.0; x < 1.1; x += 0.2) {
            cout << boost::format("poly(%5.1f) = %6.3f\n") % x % chebyFunc(x);
        }
    }
}
