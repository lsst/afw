#include <iostream>

#include <boost/format.hpp>

#include "lsst/afw/math.h"

/**
 * Demonstrate an AnalyticKernel, both spatially invariant and spatially varying.
 */

typedef lsst::afw::math::Kernel::PixelT PixelT;

using namespace std;

int main() {
    double sigmaX = 2.0;
    double sigmaY = 2.5;
    unsigned int kernelCols = 6;
    unsigned int kernelRows = 5;

    lsst::afw::math::GaussianFunction2<PixelT> gaussFunc(sigmaX, sigmaY);
    lsst::afw::math::AnalyticKernel gaussKernel(gaussFunc, kernelCols, kernelRows);
    
    cout << boost::format("Gaussian Kernel with sigmaX=%.1f, sigmaY=%.1f\n\n") % sigmaX % sigmaY;
    
    lsst::afw::math::printKernel(gaussKernel, true);
    
    // now show a spatially varying version
    unsigned int polyOrder = 1;
    lsst::afw::math::PolynomialFunction2<PixelT> polyFunc(polyOrder);

    lsst::afw::math::AnalyticKernel gaussSpVarKernel(gaussFunc, kernelCols, kernelRows, polyFunc);

    // get copy of spatial parameters (all zeros), set and feed back to the kernel
    vector<vector<double> > polyParams = gaussSpVarKernel.getSpatialParameters();
    polyParams[0][0] =  1.0;
    polyParams[0][1] =  1.0;
    polyParams[0][2] =  0.0;
    polyParams[1][0] =  1.0;
    polyParams[1][1] =  0.0;
    polyParams[1][2] =  1.0;
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

    std::vector<double> kernelParams(gaussSpVarKernel.getNKernelParameters());
    for (unsigned int y = 0; y < 2; ++y) {
        for (unsigned int x=0; x < 2; ++x) {
            gaussSpVarKernel.computeKernelParametersFromSpatialModel(
                kernelParams, static_cast<double>(x), static_cast<double>(y));
            cout << boost::format("GaussianKernel at x=%d, y=%d; xSigma = %7.2f, ySigma=%7.2f:\n\n")
                % x % y % kernelParams[0] % kernelParams[1];

            lsst::afw::math::printKernel(
                gaussSpVarKernel, true, static_cast<double>(x), static_cast<double>(y));
        }
    }
}
