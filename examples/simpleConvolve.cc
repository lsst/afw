#include <iostream>
#include <sstream>
#include <string>

#include "lsst/daf/base.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelFunctions.h"

using namespace std;
const std::string outFile("scOut");
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

int main(int argc, char **argv) {
    typedef afwMath::Kernel::Pixel pixelType;
    unsigned int kernelCols = 6;
    unsigned int kernelRows = 5;
    
    lsst::pex::logging::Trace::setDestination(std::cout);
    lsst::pex::logging::Trace::setVerbosity("lsst.afw.kernel", 5);

    const double DefSigma = 2.0;
    
    if (argc < 2) {
        std::cerr << "Usage: simpleConvolve fitsFile [sigma]" << std::endl;
        std::cerr << "fitsFile excludes the \"_img.fits\" suffix" << std::endl;
        std::cerr << "sigma (default " << DefSigma << ") is the width of the gaussian kernel, in pixels"
            << std::endl;
        return 1;
    }
    
    { // block in which to allocate and deallocate memory
    
        double sigma = DefSigma;
        if (argc > 2) {
            std::istringstream(argv[2]) >> sigma;
        }
        
        // read in fits file
        afwImage::MaskedImage<pixelType> mImage(argv[1]);
        
        // construct kernel
        afwMath::GaussianFunction2<pixelType> gaussFunc(sigma, sigma);
        afwMath::AnalyticKernel kernel(kernelCols, kernelRows, gaussFunc);
    
        // convolve
        afwImage::MaskedImage<pixelType> resMaskedImage(mImage.getDimensions());
        afwMath::convolve(resMaskedImage, mImage, kernel, true);
    
        // write results
        resMaskedImage.writeFits(outFile);
    }

     //
     // Check for memory leaks
     //
     if (lsst::daf::base::Citizen::census(0) != 0) {
         std::cerr << "Leaked memory blocks:" << std::endl;
         lsst::daf::base::Citizen::census(std::cerr);
     }
    
}
