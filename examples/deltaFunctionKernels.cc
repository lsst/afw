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
#include <vector>

#include "boost/format.hpp"

#include "lsst/afw/math.h"

namespace afwMath = lsst::afw::math;

int main() {
    unsigned int kernelCols = 3;
    unsigned int kernelRows = 2;
    unsigned int nPixels = kernelCols * kernelRows;

    // create linear combination kernel as a set of delta function basis kernels
    afwMath::KernelList kernelList;
    {
        unsigned int ind = 0;
        for (unsigned int row = 0; row < kernelRows; ++row) {
            for (unsigned int col = 0; col < kernelCols; ++col) {
                std::cout << boost::format("Delta function kernel %3d: col=%d, row=%d\n") % ind % col % row;
                PTR(afwMath::Kernel) kernelPtr(new afwMath::DeltaFunctionKernel(kernelCols, kernelRows,
                                                               lsst::afw::geom::Point2I(col, row))
                             );
                kernelList.push_back(kernelPtr);
                ++ind;
            }
        }
    }
    std::cout << std::endl;
    std::vector<double> kernelParams(nPixels); // initial kernel parameters
    afwMath::LinearCombinationKernel deltaFunctionKernelSet(kernelList, kernelParams);

    // set various kernel parameters and print the results
    for (unsigned int ind = 0; ind < nPixels; ++ind) {
        kernelParams[ind] = 1.0;
        deltaFunctionKernelSet.setKernelParameters(kernelParams);

        std::cout << "DeltaFunctionKernelSet with kernelParams=";
        for (unsigned int ii = 0; ii < nPixels; ++ii) {
            std::cout << kernelParams[ii] << " ";
        }
        std::cout << std::endl << std::endl;
        afwMath::printKernel(deltaFunctionKernelSet, true);
    }
}
