#include <iostream>

#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>

#include <lsst/fw/FunctionLibrary.h>
#include <lsst/fw/Kernel.h>
#include <lsst/fw/KernelFunctions.h>

using namespace std;

int main() {
    typedef lsst::fw::Kernel::PixelT pixelType;

    unsigned int kernelCols = 6;
    unsigned int kernelRows = 5;
    unsigned int order = (min(kernelCols, kernelRows) - 1) / 2;

    lsst::fw::Kernel::KernelFunctionPtrType kfuncPtr(
        new lsst::fw::function::LanczosFunction2<pixelType>(order));
    lsst::fw::AnalyticKernel kernel(kfuncPtr, kernelCols, kernelRows);

    cout << boost::format("Lanczos Kernel is %d x %d; Lanczos function has %d order\n")
        % kernelCols % kernelRows % order;
    
    double deltaOff = 1.0 / 3.0;
    vector<double> offVec(2);
    for (offVec[0] = 0.0; offVec[0] < 1.01; offVec[0] += deltaOff) {
        cout << boost::format("Kernel with offset %7.3f, %7.3f\n\n") % offVec[0] % offVec[1];
        
        kernel.setKernelParameters(offVec);
        
        lsst::fw::kernel::printKernel(kernel);
    }
}
