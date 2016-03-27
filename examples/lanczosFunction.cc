/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>
#include <iomanip>

#include "lsst/afw/math/FunctionLibrary.h"

using namespace std;

int main() {
    typedef double FuncReturn;
    
    unsigned int order = 2;
    lsst::afw::math::LanczosFunction2<FuncReturn> lancFunc(order);

    double deltaX = order * 2 / 12.0;
    
    double deltaOff = deltaX / 3.0;
    
    cout << fixed << setprecision(3);
    
    vector<FuncReturn> offVec(2);
    for (offVec[0] = 0.0; offVec[0] < deltaX * 1.01; offVec[0] += deltaOff) {
        cout << "LanczosFunction2(" << order << ") with offset " << offVec[0] << ", " << offVec[1]
            << endl << endl;
        
        lancFunc.setParameters(offVec);
    
        cout << "  y \\ x";
        for (double x = -(order + deltaX); x < order + (deltaX * 1.01); x += deltaX) {
            cout << setw(7) << x;
        }
        cout << endl;
        
        for (double y = -(order + deltaX); y < order + (deltaX * 1.01); y += deltaX) {
            cout << setw(7) << y;
            for (double x = -(order + deltaX); x <= order + deltaX; x += deltaX) {
                cout << setw(7) << lancFunc(x, y);
            }
            cout << endl;
        }
        cout << endl << endl;
    }
}
