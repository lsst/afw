/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>
#include <sstream>

#include "lsst/utils/Utils.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/math.h"
#include "lsst/afw/image.h"

namespace pexLog = lsst::pex::logging;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

const std::string outImagePath("svcOut.fits");

/**
 * Demonstrate convolution with a spatially varying kernel
 *
 * The kernel is a Gaussian that varies as follows:
 * xSigma varies linearly from minSigma to maxSigma as image col goes from 0 to max
 * ySigma varies linearly from minSigma to maxSigma as image row goes from 0 to max
 */
int main(int argc, char **argv) {
    typedef afwMath::Kernel::Pixel Pixel;

    pexLog::Trace::setDestination(std::cout);
    pexLog::Trace::setVerbosity("lsst.afw.kernel", 5);

    double minSigma = 0.1;
    double maxSigma = 3.0;
    unsigned int kernelCols = 5;
    unsigned int kernelRows = 5;

    std::string inImagePath;
    if (argc < 2) {
        try {
            std::string dataDir = lsst::utils::getPackageDir("afwdata");
            inImagePath = dataDir + "/data/small.fits";
        } catch (lsst::pex::exceptions::NotFoundError) {
            std::cerr << "Usage: spatiallyVaryingConvolve [fitsFile]" << std::endl;
            std::cerr << "fitsFile is the path to a masked image" << std::endl;
            std::cerr << "\nError: setup afwdata or specify fitsFile.\n" << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        inImagePath = std::string(argv[1]);
    }
    std::cerr << "Convolving masked image " << inImagePath << std::endl;
    
    // read in fits file
    afwImage::MaskedImage<Pixel> mImage(inImagePath);
    
    // construct kernel
    afwMath::GaussianFunction2<Pixel> gaussFunc(1, 1, 0);
    unsigned int polyOrder = 1;
    afwMath::PolynomialFunction2<double> polyFunc(polyOrder);
    afwMath::AnalyticKernel gaussSpVarKernel(kernelCols, kernelRows, gaussFunc, polyFunc);

    // Get copy of spatial parameters (all zeros), set and feed back to the kernel
    std::vector<std::vector<double> > polyParams = gaussSpVarKernel.getSpatialParameters();
    // Set spatial parameters for kernel parameter 0
    polyParams[0][0] = minSigma;
    polyParams[0][1] = (maxSigma - minSigma)/static_cast<double>(mImage.getWidth());
    polyParams[0][2] = 0.0;
    // Set spatial function parameters for kernel parameter 1
    polyParams[1][0] = minSigma;
    polyParams[1][1] = 0.0;
    polyParams[1][2] = (maxSigma - minSigma)/static_cast<double>(mImage.getHeight());
    gaussSpVarKernel.setSpatialParameters(polyParams);
    
    std::cout << "Spatial Parameters:" << std::endl;
    for (unsigned int row = 0; row < polyParams.size(); ++row) {
        if (row == 0) {
            std::cout << "xSigma";
        } else {
            std::cout << "ySigma";
        }
        for (unsigned int col = 0; col < polyParams[row].size(); ++col) {
            std::cout << boost::format("%7.1f") % polyParams[row][col];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // convolve
    afwImage::MaskedImage<Pixel> resMaskedImage(mImage.getDimensions());
    afwMath::convolve(resMaskedImage, mImage, gaussSpVarKernel, true);

    // write results
    resMaskedImage.writeFits(outImagePath);
    std::cerr << "Wrote convolved image " << outImagePath << std::endl;
}
