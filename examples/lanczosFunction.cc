/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
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
