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
