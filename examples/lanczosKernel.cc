/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>

#include "boost/shared_ptr.hpp"
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

    cout << boost::format("Lanczos Kernel is %d x %d; Lanczos function has order %d\n")
        % kernelCols % kernelRows % order;
    
    double deltaOff = 1.0 / 3.0;
    vector<double> offVec(2);
    for (offVec[0] = 0.0; offVec[0] < 1.01; offVec[0] += deltaOff) {
        cout << boost::format("Kernel with offset %7.3f, %7.3f\n\n") % offVec[0] % offVec[1];
        
        kernel.setKernelParameters(offVec);
        
        lsst::afw::math::printKernel(kernel, true);
    }
}
