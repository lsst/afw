#include <iostream>
#include <sstream>
#include <string>

#include "lsst/daf/base.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelFunctions.h"

using namespace std;
const std::string outFile("scOut");
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

int main(int argc, char **argv) {
    typedef afwMath::Kernel::PixelT pixelType;
    unsigned int kernelCols = 6;
    unsigned int kernelRows = 5;
    
    lsst::pex::logging::Trace::setDestination(std::cout);
    lsst::pex::logging::Trace::setVerbosity("lsst.afw.kernel", 5);

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
        afwImage::MaskedImage<pixelType> mImage(argv[1]);
        
        // construct kernel
        afwMath::GaussianFunction2<pixelType> gaussFunc(sigma, sigma);
        afwMath::AnalyticKernel kernel(gaussFunc, kernelCols, kernelRows);
    
        // convolve
        afwImage::MaskedImage<pixelType>
            resMaskedImage = afwMath::convolveNew(mImage, kernel, edgeMaskBit, true);
    
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
