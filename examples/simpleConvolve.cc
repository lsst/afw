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
    int DefEdgeMaskBit = 0;
    
    if (argc < 2) {
        std::cerr << "Usage: simpleConvolve fitsFile [sigma [edgeMaskBit]]" << std::endl;
        std::cerr << "fitsFile excludes the \"_img.fits\" suffix" << std::endl;
        std::cerr << "sigma (default " << DefSigma << ") is the width of the gaussian kernel, in pixels"
            << std::endl;
        std::cerr << "edgeMaskBit (default " << DefEdgeMaskBit
            << ") bit to set around the edge (none if < 0)" << std::endl;
        return 1;
    }
    
    { // block in which to allocate and deallocate memory
    
        double sigma = DefSigma;
        if (argc > 2) {
            std::istringstream(argv[2]) >> sigma;
        }
        
        int edgeMaskBit = DefEdgeMaskBit;
        if (argc > 3) {
            std::istringstream(argv[3]) >> edgeMaskBit;
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
            resMaskedImage = lsst::fw::kernel::convolve(mImage, kernel, edgeMaskBit, true);
    
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
