#include <iostream>
#include <sstream>
#include <ctime>

#include <lsst/fw/FunctionLibrary.h>
#include <lsst/fw/Image.h>
#include <lsst/fw/MaskedImage.h>
#include <lsst/fw/Kernel.h>
#include <lsst/fw/KernelFunctions.h>

using namespace std;

/**
 * Time convolution with a spatially varying kernel
 *
 * The kernel is a Gaussian that varies as follows:
 * xSigma varies linearly from minSigma to maxSigma as image col goes from 0 to max
 * ySigma varies linearly from minSigma to maxSigma as image row goes from 0 to max
 */
int main(int argc, char **argv) {
    typedef float pixelType;
    typedef unsigned int maskType;
    double minSigma = 0.1;
    double maxSigma = 3.0;
    const unsigned int DefNIter = 10;
    const unsigned int MinKernelSize = 5;
    const unsigned int MaxKernelSize = 15;
    const unsigned int DeltaKernelSize = 5;

    if (argc < 2) {
        std::cout << "Usage: timeSpatiallyVaryingConvolve fitsFile [nIter [threshold]]" << std::endl;
        std::cerr << "fitsFile excludes the \"_img.fits\" suffix" << std::endl;
        std::cout << "nIter (default " << DefNIter << ") is the number of iterations per kernel size" << endl;
        std::cout << "Kernel size ranges from " << MinKernelSize << " to " << MaxKernelSize
            << " in steps of " << DeltaKernelSize << " pixels on a side" << endl;
        return 1;
    }
    
    unsigned int nIter = 10;
    if (argc > 2) {
        istringstream(argv[2]) >> nIter;
    }
    
    pixelType threshold = 1000;
    if (argc > 3) {
        istringstream(argv[3]) >> threshold;
    }
    
    // read in fits file
    lsst::fw::MaskedImage<pixelType, maskType> mImage;
    mImage.readFits(argv[1]);
    
    cout << "Image is " << mImage.getCols() << " by " << mImage.getRows() << endl;
    
    for (unsigned int kSize = MinKernelSize; kSize <= MaxKernelSize; kSize += DeltaKernelSize) {
        // construct kernel
        lsst::fw::Kernel<pixelType>::KernelFunctionPtrType gaussFuncPtr(
            new lsst::fw::function::GaussianFunction2<pixelType>(1, 1));
        unsigned int polyOrder = 1;
        lsst::fw::Kernel<pixelType>::SpatialFunctionPtrType polyFuncPtr(
            new lsst::fw::function::PolynomialFunction2<double>(polyOrder));
        lsst::fw::AnalyticKernel<pixelType> gaussSpVarKernel(
            gaussFuncPtr, kSize, kSize, polyFuncPtr);
    
        // get copy of spatial parameters (all zeros), set and feed back to the kernel
        vector<vector<double> > polyParams = gaussSpVarKernel.getSpatialParameters();
        polyParams[0][0] = minSigma;
        polyParams[0][1] = (maxSigma - minSigma) / static_cast<double>(mImage.getCols());
        polyParams[0][2] = 0.0;
        polyParams[1][0] = minSigma;
        polyParams[1][1] = 0.0;
        polyParams[1][2] = (maxSigma - minSigma) / static_cast<double>(mImage.getRows());
        gaussSpVarKernel.setSpatialParameters(polyParams);
        
        clock_t startTime = clock();
        for (unsigned int iter = 0; iter < nIter; ++iter) {
            // convolve
            lsst::fw::MaskedImage<pixelType, maskType>
                resMImage = lsst::fw::kernel::convolve(mImage, gaussSpVarKernel, threshold, -1);
        }
        double secPerIter = (clock() - startTime) / static_cast<double> (nIter * CLOCKS_PER_SEC);
        cout << secPerIter << " sec/convolution for a " << kSize << " by " << kSize << " kernel" << endl;
    }
}
