/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>
#include <vector>

#include "boost/shared_ptr.hpp"
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
