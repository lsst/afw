#include <ctime>
#include <iostream>
#include <sstream>
#include <vector>

#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelFunctions.h"

/**
 * Time convolution with a spatially varying kernel
 *
 * The kernel is a Gaussian that varies as follows:
 * xSigma varies linearly from minSigma to maxSigma as image col goes from 0 to max
 * ySigma varies linearly from minSigma to maxSigma as image row goes from 0 to max
 */
int main(int argc, char **argv) {
    typedef float imageType;
    typedef double kernelType;
    double minSigma = 0.1;
    double maxSigma = 3.0;
    const int EdgeMaskBit = -1;
    const unsigned int DefNIter = 10;
    const unsigned int MinKernelSize = 5;
    const unsigned int MaxKernelSize = 15;
    const unsigned int DeltaKernelSize = 5;

    if (argc < 2) {
        std::cout << "Usage: timeSpatiallyVaryingConvolve fitsFile [nIter]" << std::endl;
        std::cout << "fitsFile excludes the \"_img.fits\" suffix" << std::endl;
        std::cout << "nIter (default " << DefNIter << ") is the number of iterations per kernel size" << std::endl;
        std::cout << "Kernel size ranges from " << MinKernelSize << " to " << MaxKernelSize
            << " in steps of " << DeltaKernelSize << " pixels on a side" << std::endl;
        return 1;
    }
    
    unsigned int nIter = DefNIter;
    if (argc > 2) {
        std::istringstream(argv[2]) >> nIter;
    }
    
    // read in fits file
    lsst::afw::image::MaskedImage<imageType, lsst::afw::image::maskPixelType> mImage;
    mImage.readFits(argv[1]);
    
    std::cout << "Image is " << mImage.getCols() << " by " << mImage.getRows() << std::endl;
    
    for (unsigned int kSize = MinKernelSize; kSize <= MaxKernelSize; kSize += DeltaKernelSize) {
        // construct kernel
        lsst::afw::math::GaussianFunction2<kernelType> gaussFunc(1, 1);
        unsigned int polyOrder = 1;
        lsst::afw::math::PolynomialFunction2<double> polyFunc(polyOrder);
        lsst::afw::math::AnalyticKernel gaussSpVarKernel(gaussFunc, kSize, kSize, polyFunc);
    
        // get copy of spatial parameters (all zeros), set and feed back to the kernel
        std::vector<std::vector<double> > polyParams = gaussSpVarKernel.getSpatialParameters();
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
            lsst::afw::image::MaskedImage<imageType, lsst::afw::image::maskPixelType>
                resMImage = lsst::afw::math::convolve(mImage, gaussSpVarKernel, EdgeMaskBit, true);
        }
        double secPerIter = (clock() - startTime) / static_cast<double> (nIter * CLOCKS_PER_SEC);
        std::cout << secPerIter << " sec/convolution for a " << kSize << " by " << kSize << " kernel" << std::endl;
    }
}
