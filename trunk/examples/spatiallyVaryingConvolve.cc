/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
#include <iostream>
#include <sstream>

#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelFunctions.h"

using namespace std;
namespace pexLog = lsst::pex::logging;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

const std::string outFile("svcOut");

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

    std::string mimg;
    if (argc < 2) {
        std::string afwdata = getenv("AFWDATA_DIR");
        if (afwdata.empty()) {
            std::cerr << "Usage: simpleConvolve fitsFile" << std::endl;
            std::cerr << "fitsFile excludes the \"_img.fits\" suffix" << std::endl;
            std::cerr << "I can take a default file from AFWDATA_DIR, but it's not defined." << std::endl;
            std::cerr << "Is afwdata set up?\n" << std::endl;
            exit(EXIT_FAILURE);
        } else {
            mimg = afwdata + "/small_MI";
            std::cerr << "Using " << mimg << std::endl;
        }
    } else {
        mimg = std::string(argv[1]);
    }
    
    // read in fits file
    afwImage::MaskedImage<Pixel> mImage(mimg);
    
    // construct kernel
    afwMath::GaussianFunction2<Pixel> gaussFunc(1, 1, 0);
    unsigned int polyOrder = 1;
    afwMath::PolynomialFunction2<double> polyFunc(polyOrder);
    afwMath::AnalyticKernel gaussSpVarKernel(kernelCols, kernelRows, gaussFunc, polyFunc);

    // Get copy of spatial parameters (all zeros), set and feed back to the kernel
    vector<vector<double> > polyParams = gaussSpVarKernel.getSpatialParameters();
    // Set spatial parameters for kernel parameter 0
    polyParams[0][0] = minSigma;
    polyParams[0][1] = (maxSigma - minSigma)/static_cast<double>(mImage.getWidth());
    polyParams[0][2] = 0.0;
    // Set spatial function parameters for kernel parameter 1
    polyParams[1][0] = minSigma;
    polyParams[1][1] = 0.0;
    polyParams[1][2] = (maxSigma - minSigma)/static_cast<double>(mImage.getHeight());
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
    afwImage::MaskedImage<Pixel> resMaskedImage(mImage.getDimensions());
    afwMath::convolve(resMaskedImage, mImage, gaussSpVarKernel, true);

    // write results
    resMaskedImage.writeFits(outFile);
}
