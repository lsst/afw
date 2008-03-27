#include <iostream>
#include <iomanip>

#include <lsst/afw/math/FunctionLibrary.h>

using namespace std;

int main() {
    typedef float funcType;
    
    funcType xo = 1.0, yo = -2.0;
    
    lsst::afw::math::IntegerDeltaFunction2<funcType> deltaFunc(xo, yo);
    
    cout << "IntegerDeltaFunction2(" << xo << ", " << yo << ")" << endl;

    cout << " y\\x";
    for (double x = -3; x <= 3; ++x) {
        cout << setw(4) << x;
    }
    cout << endl;
    
    for (double y = -3; y <= 3; ++y) {
        cout << setw(4) << y;
        for (double x = -3; x <= 3; ++x) {
            cout << setw(4) << deltaFunc(x, y);
        }
        cout << endl;
    }
    cout << endl;
}
