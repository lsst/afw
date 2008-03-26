#include <iostream>

#include <boost/format.hpp>

#include <lsst/fw/FunctionLibrary.h>

using namespace std;

int main() {
    typedef double funcType;
    double sigma = 2.0;
    lsst::fw::function::GaussianFunction1<funcType> gaussFunc(sigma);

    for (double x = -3.0; x < 3.1; x += 0.5) {
        cout << boost::format("gauss(%5.1f) = %6.3f\n") % x % gaussFunc(x);
    }
}
