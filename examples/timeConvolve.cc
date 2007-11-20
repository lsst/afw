#include <iostream>
#include <sstream>
#include <ctime>

#include <lsst/fw/FunctionLibrary.h>
#include <lsst/fw/Image.h>
#include <lsst/fw/MaskedImage.h>
#include <lsst/fw/Kernel.h>
#include <lsst/fw/KernelFunctions.h>

using namespace std;

int main(int argc, char **argv) {
    typedef double pixelType;
    double sigma = 3;
    const int EdgeMaskBit = -1;
    const unsigned DefNIter = 10;
    const unsigned MinKernelSize = 5;
    const unsigned MaxKernelSize = 15;
    const unsigned DeltaKernelSize = 5;

    if (argc < 2) {
        std::cout << "Usage: timeConvolve fitsFile [nIter]" << std::endl;
        std::cerr << "fitsFile excludes the \"_img.fits\" suffix" << std::endl;
        std::cout << "nIter (default " << DefNIter << ") is the number of iterations per kernel size" << endl;
        std::cout << "Kernel size ranges from " << MinKernelSize << " to " << MaxKernelSize
            << " in steps of " << DeltaKernelSize << " pixels on a side" << endl;
        return 1;
    }
    
    unsigned nIter = 10;
    if (argc > 2) {
        istringstream(argv[2]) >> nIter;
    }
    
    // read in fits file
    lsst::fw::MaskedImage<pixelType, lsst::fw::maskPixelType> mImage;
    mImage.readFits(argv[1]);
    
    cout << "Image is " << mImage.getCols()
        << " by " << mImage.getRows() << endl;
    
    for (unsigned kSize = MinKernelSize; kSize <= MaxKernelSize; kSize += DeltaKernelSize) {
        // construct kernel
        lsst::fw::Kernel<pixelType>::KernelFunctionPtrType kfuncPtr(
            new lsst::fw::function::GaussianFunction2<pixelType>(sigma, sigma));
        lsst::fw::AnalyticKernel<pixelType> kernel(kfuncPtr, kSize, kSize);
        
        clock_t startTime = clock();
        for (unsigned iter = 0; iter < nIter; ++iter) {
            // convolve
            lsst::fw::MaskedImage<pixelType, lsst::fw::maskPixelType>
                resMImage = lsst::fw::kernel::convolve(mImage, kernel, EdgeMaskBit, true);
        }
        double secPerIter = (clock() - startTime) / static_cast<double> (nIter * CLOCKS_PER_SEC);
        cout << secPerIter << " sec/convolution for a " << kSize << " by " << kSize << " kernel" << endl;
    }
}
