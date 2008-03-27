#include <iostream>
#include <vector>

#include <boost/format.hpp>

#include <lsst/afw/math/FunctionLibrary.h>

using namespace std;

int main() {
    typedef double funcType;
    const unsigned int order = 2;
    vector<double> params(order+1);
    lsst::afw::math::PolynomialFunction1<funcType> polyFunc(order);

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
