#include <iostream>
#include <sstream>

#include <lsst/fw/FunctionLibrary.h>
#include <lsst/fw/Image.h>
#include <lsst/fw/MaskedImage.h>
#include <lsst/mwi/utils/Trace.h>
#include <lsst/fw/Kernel.h>
#include <lsst/fw/KernelFunctions.h>

using namespace std;
namespace mwiu = lsst::mwi::utils;

const std::string outFile("scOut");

int main(int argc, char **argv) {
    mwiu::Trace::setDestination(std::cout);
    mwiu::Trace::setVerbosity("lsst.fw.kernel", 5);

    typedef float pixelType;
    typedef unsigned int maskType;
    const double DefSigma = 2.0;
    const pixelType DefThreshold = 0.1;

    if (argc < 2) {
        std::cerr << "Usage: simpleConvolve fitsFile [sigma [threshold]]" << std::endl;
        std::cerr << "fitsFile excludes the \"_img.fits\" suffix" << std::endl;
        std::cerr << "sigma (default " << DefSigma << ") is the width of the gaussian kernel, in pixels"
            << std::endl;
        std::cerr << "threshold (default " << DefThreshold
            << ") is the kernel value above which a bad pixel is significant" << std::endl;
        return 1;
    }
    
    double sigma = DefSigma;
    if (argc > 2) {
        istringstream(argv[2]) >> sigma;
    }
    
    pixelType threshold = DefThreshold;
    if (argc > 3) {
        istringstream(argv[3]) >> threshold;
    }
    
    // read in fits file
    lsst::fw::MaskedImage<pixelType, maskType> mImage;
    mImage.readFits(argv[1]);
    
    // construct kernel
    lsst::fw::Kernel<pixelType>::KernelFunctionPtrType kfuncPtr(
        new lsst::fw::function::GaussianFunction2<pixelType>(sigma, sigma));
    lsst::fw::AnalyticKernel<pixelType> kernel(kfuncPtr, 5, 5);

    // convolve
    lsst::fw::MaskedImage<pixelType, maskType>
        resMaskedImage = lsst::fw::kernel::convolve(mImage, kernel, threshold, 0);

    // write results
    resMaskedImage.writeFits(outFile);
}
