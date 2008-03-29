#include <iostream>
#include <sstream>

#include <lsst/afw/math/FunctionLibrary.h>
#include <lsst/afw/image/Image.h>
#include <lsst/afw/image/MaskedImage.h>
#include <lsst/pex/logging/Trace.h>
#include <lsst/afw/math/Kernel.h>
#include <lsst/afw/math/KernelFunctions.h>

using namespace std;
namespace pexLog = lsst::pex::logging;

const std::string outFile("svcOut");

/**
 * Demonstrate convolution with a spatially varying kernel
 *
 * The kernel is a Gaussian that varies as follows:
 * xSigma varies linearly from minSigma to maxSigma as image col goes from 0 to max
 * ySigma varies linearly from minSigma to maxSigma as image row goes from 0 to max
 */
int main(int argc, char **argv) {
    typedef double pixelType;

    pexLog::Trace::setDestination(std::cout);
    pexLog::Trace::setVerbosity("lsst.afw.kernel", 5);

    double minSigma = 0.1;
    double maxSigma = 3.0;
    unsigned int kernelCols = 5;
    unsigned int kernelRows = 5;
    const int DefEdgeMaskBit = 15;

    if (argc < 2) {
        std::cerr << "Usage: simpleConvolve fitsFile [edgeMaskBit]" << std::endl;
        std::cerr << "fitsFile excludes the \"_img.fits\" suffix" << std::endl;
        std::cerr << "edgeMaskBit (default " << DefEdgeMaskBit
            << ")  is the edge-extended mask bit (-1 to disable)" << std::endl;
        return 1;
    }
    
    int edgeMaskBit = DefEdgeMaskBit;
    if (argc > 2) {
        istringstream(argv[2]) >> edgeMaskBit;
    }
    
    // read in fits file
    lsst::afw::image::MaskedImage<pixelType, lsst::afw::image::maskPixelType> mImage;
    mImage.readFits(argv[1]);
    
    // construct kernel
    lsst::afw::math::Kernel::KernelFunctionPtrType gaussFuncPtr(
        new lsst::afw::math::GaussianFunction2<pixelType>(1, 1));
    unsigned int polyOrder = 1;
    lsst::afw::math::Kernel::SpatialFunctionPtrType polyFuncPtr(
        new lsst::afw::math::PolynomialFunction2<double>(polyOrder));
    lsst::afw::math::AnalyticKernel gaussSpVarKernel(
        gaussFuncPtr, kernelCols, kernelRows, polyFuncPtr);

    // Get copy of spatial parameters (all zeros), set and feed back to the kernel
    vector<vector<double> > polyParams = gaussSpVarKernel.getSpatialParameters();
    // Set spatial parameters for kernel parameter 0
    polyParams[0][0] = minSigma;
    polyParams[0][1] = (maxSigma - minSigma) / static_cast<double>(mImage.getCols());
    polyParams[0][2] = 0.0;
    // Set spatial function parameters for kernel parameter 1
    polyParams[1][0] = minSigma;
    polyParams[1][1] = 0.0;
    polyParams[1][2] = (maxSigma - minSigma) / static_cast<double>(mImage.getRows());
    gaussSpVarKernel.setSpatialParameters(polyParams);
    
    cout << "Spatial Parameters:" << endl;
    for (unsigned int row = 0; row < polyParams.size(); ++row) {
        if (row == 0) {
            cout << "xSigma";
        } else {
            cout << "ySigma";
        }
        for (unsigned int col = 0; col < polyParams[row].size(); ++col) {
            cout << boost::format("%7.1f") % polyParams[row][col];
        }
        cout << endl;
    }
    cout << endl;

    // convolve
    lsst::afw::image::MaskedImage<pixelType, lsst::afw::image::maskPixelType>
        resMaskedImage = lsst::afw::math::convolve(mImage, gaussSpVarKernel, edgeMaskBit, true);

    // write results
    resMaskedImage.writeFits(outFile);
}
