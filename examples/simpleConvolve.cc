/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>
#include <sstream>
#include <string>

#include "lsst/utils/Utils.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/math.h"
#include "lsst/afw/image.h"

const std::string outImagePath("scOut.fits");
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

int main(int argc, char **argv) {
    typedef afwMath::Kernel::Pixel Pixel;
    unsigned int kernelCols = 6;
    unsigned int kernelRows = 5;
    
    lsst::pex::logging::Trace::setDestination(std::cout);
    lsst::pex::logging::Trace::setVerbosity("lsst.afw.kernel", 5);

    const double DefSigma = 2.0;
    
    std::string inImagePath;
    if (argc < 2) {
        try {
            std::string dataDir = lsst::utils::getPackageDir("afwdata");
            inImagePath = dataDir + "/data/small.fits";
        } catch (lsst::pex::exceptions::NotFoundError) {
            std::cerr << "Usage: simpleConvolve [fitsFile [sigma]]" << std::endl;
            std::cerr << "fitsFile is the path to a masked image" << std::endl;
            std::cerr << "sigma (default " << DefSigma << ") is the width of the gaussian kernel, in pixels"
                      << std::endl;
            std::cerr << "\nError: setup afwdata or specify fitsFile.\n" << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        inImagePath = std::string(argv[1]);
    }
    std::cerr << "Convolving masked image " << inImagePath << std::endl;
    
    double sigma = DefSigma;
    if (argc > 2) {
        std::istringstream(argv[2]) >> sigma;
    }
    
    // read in fits file
    afwImage::MaskedImage<Pixel> mImage(inImagePath);
    
    // construct kernel
    afwMath::GaussianFunction2<Pixel> gaussFunc(sigma, sigma, 0);
    afwMath::AnalyticKernel kernel(kernelCols, kernelRows, gaussFunc);

    // convolve
    afwImage::MaskedImage<Pixel> resMaskedImage(mImage.getDimensions());
    afwMath::convolve(resMaskedImage, mImage, kernel, true);

    // write results
    resMaskedImage.writeFits(outImagePath);
    std::cerr << "Wrote convolved image " << outImagePath << std::endl;
}
