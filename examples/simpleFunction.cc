#include <iostream>

#include <lsst/fw/FunctionLibrary.h>

using namespace std;

int main() {
    lsst::fw::GaussianFunction1<double> gaussFunc(1.0, 2.0); // ampl, sigma

    for (double x = -3.0; x < 3.1; x += 0.5) {
        cout << "gauss(" << x << ") = " << gaussFunc(x) << endl;
    }
}
