#include <iostream>
#include <sstream>
#include <string>

#include <lsst/daf/data/Citizen.h>
#include <lsst/pex/utils/Trace.h>
#include <lsst/afw/math/FunctionLibrary.h>
#include <lsst/afw/image/Image.h>
#include <lsst/afw/math/Kernel.h>
#include <lsst/afw/math/KernelFunctions.h>

using namespace std;
const std::string outFile("scOut");

int main(int argc, char **argv) {
    typedef lsst::afw::math::Kernel::PixelT pixelType;
    
    lsst::pex::utils::Trace::setDestination(std::cout);
    lsst::pex::utils::Trace::setVerbosity("lsst.fw.kernel", 5);

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
        lsst::afw::image::MaskedImage<pixelType, lsst::afw::maskPixelType> mImage;
        mImage.readFits(argv[1]);
        
        // construct kernel
        lsst::afw::math::Kernel::KernelFunctionPtrType kfuncPtr(
            new lsst::afw::math::GaussianFunction2<pixelType>(sigma, sigma));
        lsst::afw::math::AnalyticKernel kernel(kfuncPtr, 5, 5);
    
        // convolve
        lsst::afw::image::MaskedImage<pixelType, lsst::afw::maskPixelType>
            resMaskedImage = lsst::afw::math::convolve(mImage, kernel, edgeMaskBit, true);
    
        // write results
        resMaskedImage.writeFits(outFile);
    }

     //
     // Check for memory leaks
     //
     if (lsst::daf::data::Citizen::census(0) != 0) {
         std::cerr << "Leaked memory blocks:" << std::endl;
         lsst::daf::data::Citizen::census(std::cerr);
     }
    
}
