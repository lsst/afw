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

const std::string outFile("svcOut");

/**
 * Demonstrate convolution with a spatially varying kernel
 *
 * The kernel is a Gaussian that varies as follows:
 * xSigma varies linearly from minSigma to maxSigma as image col goes from 0 to max
 * ySigma varies linearly from minSigma to maxSigma as image row goes from 0 to max
 */
int main(int argc, char **argv) {
    mwiu::Trace::setDestination(std::cout);
    mwiu::Trace::setVerbosity("lsst.fw.kernel", 5);

    typedef float pixelType;
    typedef unsigned int maskType;
    double minSigma = 0.1;
    double maxSigma = 3.0;
    unsigned int kernelCols = 5;
    unsigned int kernelRows = 5;
    const pixelType DefThreshold = 0.1;
    const int DefEdgeMaskBit = 15;

    if (argc < 2) {
        std::cerr << "Usage: simpleConvolve fitsFile [edgeMaskBit [threshold]]" << std::endl;
        std::cerr << "fitsFile excludes the \"_img.fits\" suffix" << std::endl;
        std::cerr << "edgeMaskBit (default " << DefEdgeMaskBit
            << ")  is the edge-extended mask bit (-1 to disable)" << std::endl;
        std::cerr << "threshold (default " << DefThreshold
            << ") is the kernel value above which a bad pixel is significant" << std::endl;
        return 1;
    }
    
    int edgeMaskBit = DefEdgeMaskBit;
    if (argc > 2) {
        istringstream(argv[2]) >> edgeMaskBit;
    }
    
    pixelType threshold = DefThreshold;
    if (argc > 3) {
        istringstream(argv[3]) >> threshold;
    }
    
    // read in fits file
    lsst::fw::MaskedImage<pixelType, maskType> mImage;
    mImage.readFits(argv[1]);
    
    // construct kernel
    lsst::fw::Kernel<pixelType>::KernelFunctionPtrType gaussFuncPtr(
        new lsst::fw::function::GaussianFunction2<pixelType>(1, 1));
    unsigned int polyOrder = 1;
    lsst::fw::Kernel<pixelType>::SpatialFunctionPtrType polyFuncPtr(
        new lsst::fw::function::PolynomialFunction2<double>(polyOrder));
    lsst::fw::AnalyticKernel<pixelType> gaussSpVarKernel(
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
    lsst::fw::MaskedImage<pixelType, maskType>
        resMaskedImage = lsst::fw::kernel::convolve(mImage, gaussSpVarKernel, threshold, edgeMaskBit);

    // write results
    resMaskedImage.writeFits(outFile);
}
