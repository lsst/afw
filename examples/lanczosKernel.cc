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

#include <memory>
#include "boost/format.hpp"

#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelFunctions.h"

using namespace std;

int main() {
    typedef lsst::afw::math::Kernel::Pixel Pixel;

    unsigned int kernelCols = 6;
    unsigned int kernelRows = 5;
    unsigned int order = (min(kernelCols, kernelRows) - 1) / 2;

    lsst::afw::math::LanczosFunction2<Pixel> lanczosFunc(order);
    lsst::afw::math::AnalyticKernel kernel(kernelCols, kernelRows, lanczosFunc);

    cout << boost::format("Lanczos Kernel is %d x %d; Lanczos function has order %d\n") % kernelCols %
                    kernelRows % order;

    double deltaOff = 1.0 / 3.0;
    vector<double> offVec(2);
    for (offVec[0] = 0.0; offVec[0] < 1.01; offVec[0] += deltaOff) {
        cout << boost::format("Kernel with offset %7.3f, %7.3f\n\n") % offVec[0] % offVec[1];

        kernel.setKernelParameters(offVec);

        lsst::afw::math::printKernel(kernel, true);
    }
}
