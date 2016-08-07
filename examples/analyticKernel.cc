// -*- lsst-c++ -*-

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

#include "boost/format.hpp"

#include "lsst/afw/math.h"

/**
 * Demonstrate an AnalyticKernel, both spatially invariant and spatially varying.
 */

typedef lsst::afw::math::Kernel::Pixel Pixel;

using namespace std;

int main() {
    double majorSigma = 2.5;
    double minorSigma = 2.0;
    double angle = 0.5;
    unsigned int kernelCols = 6;
    unsigned int kernelRows = 5;

    lsst::afw::math::GaussianFunction2<Pixel> gaussFunc(majorSigma, minorSigma, angle);
    lsst::afw::math::AnalyticKernel gaussKernel(kernelCols, kernelRows, gaussFunc);

    cout << boost::format("Gaussian Kernel with majorSigma=%.1f, minorSigma=%.1f\n\n") %
        majorSigma % minorSigma;

    lsst::afw::math::printKernel(gaussKernel, true);

    // now show a spatially varying version
    unsigned int polyOrder = 1;
    lsst::afw::math::PolynomialFunction2<Pixel> polyFunc(polyOrder);

    lsst::afw::math::AnalyticKernel gaussSpVarKernel(kernelCols, kernelRows, gaussFunc, polyFunc);

    // get copy of spatial parameters (all zeros), set and feed back to the kernel
    vector<vector<double> > polyParams = gaussSpVarKernel.getSpatialParameters();
    polyParams[0][0] =  1.0;
    polyParams[0][1] =  1.0;
    polyParams[0][2] =  0.0;
    polyParams[1][0] =  1.0;
    polyParams[1][1] =  0.0;
    polyParams[1][2] =  1.0;
    gaussSpVarKernel.setSpatialParameters(polyParams);

    cout << "Spatial Parameters:" << endl;
    for (unsigned int row = 0; row < polyParams.size(); ++row) {
        if (row == 0) {
            cout << "xSigma";
        } else {
            cout << "ySigma";
        }
        for (unsigned int col = 0; col < polyParams[row].size(); ++col) {
            cout << boost::format("%7.1f") % polyParams[row][col];
        }
        cout << endl;
    }
    cout << endl;

    std::vector<double> kernelParams(gaussSpVarKernel.getNKernelParameters());
    for (unsigned int y = 0; y < 2; ++y) {
        for (unsigned int x=0; x < 2; ++x) {
            gaussSpVarKernel.computeKernelParametersFromSpatialModel(
                kernelParams, static_cast<double>(x), static_cast<double>(y));
            cout << boost::format("GaussianKernel at x=%d, y=%d; xSigma = %7.2f, ySigma=%7.2f:\n\n")
                % x % y % kernelParams[0] % kernelParams[1];

            lsst::afw::math::printKernel(
                gaussSpVarKernel, true, static_cast<double>(x), static_cast<double>(y));
        }
    }
}
