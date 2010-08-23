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
    typedef afwMath::Kernel::Pixel Pixel;
    unsigned int kernelCols = 6;
    unsigned int kernelRows = 5;
    
    lsst::pex::logging::Trace::setDestination(std::cout);
    lsst::pex::logging::Trace::setVerbosity("lsst.afw.kernel", 5);

    const double DefSigma = 2.0;
    
    std::string mimg;
    if (argc < 2) {
        std::string afwdata = getenv("AFWDATA_DIR");
        if (afwdata.empty()) {
            std::cerr << "Usage: simpleConvolve fitsFile [sigma]" << std::endl;
            std::cerr << "fitsFile excludes the \"_img.fits\" suffix" << std::endl;
            std::cerr << "sigma (default " << DefSigma << ") is the width of the gaussian kernel, in pixels"
                      << std::endl;
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
    
    { // block in which to allocate and deallocate memory
    
        double sigma = DefSigma;
        if (argc > 2) {
            std::istringstream(argv[2]) >> sigma;
        }
        
        // read in fits file
        afwImage::MaskedImage<Pixel> mImage(mimg);
        
        // construct kernel
        afwMath::GaussianFunction2<Pixel> gaussFunc(sigma, sigma, 0);
        afwMath::AnalyticKernel kernel(kernelCols, kernelRows, gaussFunc);
    
        // convolve
        afwImage::MaskedImage<Pixel> resMaskedImage(mImage.getDimensions());
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
