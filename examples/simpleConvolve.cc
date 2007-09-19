#include <iostream>
#include <sstream>
#include <string>

#include <lsst/mwi/data/Citizen.h>
#include <lsst/mwi/utils/Trace.h>
#include <lsst/fw/FunctionLibrary.h>
#include <lsst/fw/Image.h>
#include <lsst/fw/Kernel.h>
#include <lsst/fw/KernelFunctions.h>

using namespace std;
const std::string outFile("scOut");

int main(int argc, char **argv) {
    lsst::mwi::utils::Trace::setDestination(std::cout);
    lsst::mwi::utils::Trace::setVerbosity("lsst.fw.kernel", 5);

    typedef double pixelType;
    const double DefSigma = 2.0;
    const pixelType DefThreshold = 0.1;
    int DefEdgeBit = 0;
    
    if (argc < 2) {
        std::cerr << "Usage: simpleConvolve fitsFile [sigma [threshold]]" << std::endl;
        std::cerr << "fitsFile excludes the \"_img.fits\" suffix" << std::endl;
        std::cerr << "sigma (default " << DefSigma << ") is the width of the gaussian kernel, in pixels"
            << std::endl;
        std::cerr << "threshold (default " << DefThreshold
            << ") is the kernel value above which a bad pixel is significant" << std::endl;
        std::cerr << "edgeBit (default " << DefEdgeBit
            << ") bit to set around the edge (none if < 0)" << std::endl;
        return 1;
    }
    
    { // block in which to allocate and deallocate memory
    
        double sigma = DefSigma;
        if (argc > 2) {
            std::istringstream(argv[2]) >> sigma;
        }
        
        pixelType threshold = DefThreshold;
        if (argc > 3) {
            std::istringstream(argv[3]) >> threshold;
        }

        int edgeBit = DefEdgeBit;
        if (argc > 4) {
            std::istringstream(argv[4]) >> edgeBit;
        }
        
        // read in fits file
        lsst::fw::MaskedImage<pixelType, lsst::fw::maskPixelType> mImage;
        mImage.readFits(argv[1]);
        
        // construct kernel
        lsst::fw::Kernel<pixelType>::KernelFunctionPtrType kfuncPtr(
            new lsst::fw::function::GaussianFunction2<pixelType>(sigma, sigma));
        lsst::fw::AnalyticKernel<pixelType> kernel(kfuncPtr, 5, 5);
    
        // convolve
        lsst::fw::MaskedImage<pixelType, lsst::fw::maskPixelType>
            resMaskedImage = lsst::fw::kernel::convolve(mImage, kernel, threshold, edgeBit);
    
        // write results
        resMaskedImage.writeFits(outFile);
    }

     //
     // Check for memory leaks
     //
     if (lsst::mwi::data::Citizen::census(0) != 0) {
         std::cerr << "Leaked memory blocks:" << std::endl;
         lsst::mwi::data::Citizen::census(std::cerr);
     }
    
}
