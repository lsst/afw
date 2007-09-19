#include <iostream>

#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>

#include <lsst/fw/FunctionLibrary.h>
#include <lsst/fw/Kernel.h>
#include <lsst/fw/KernelFunctions.h>

using namespace std;

int main() {
    typedef double pixelType;
    
    double sigmaX = 2.0;
    double sigmaY = 2.5;
    unsigned int kernelCols = 5;
    unsigned int kernelRows = 4;

    lsst::fw::Kernel<pixelType>::KernelFunctionPtrType kfuncPtr(
        new lsst::fw::function::GaussianFunction2<pixelType>(sigmaX, sigmaY));
    lsst::fw::AnalyticKernel<pixelType> analyticKernel(kfuncPtr, kernelCols, kernelRows);
    pixelType imSum;
    lsst::fw::Image<pixelType> analyticImage = analyticKernel.computeNewImage(imSum);
    analyticImage *= 47.3; // denormalize by some arbitrary factor
    
    lsst::fw::FixedKernel<pixelType> fixedKernel(analyticImage);

    cout << boost::format("Gaussian kernel with sigmaX=%.1f, sigmaY=%.1f\n") % sigmaX % sigmaY;

    lsst::fw::kernel::printKernel(fixedKernel);
}
