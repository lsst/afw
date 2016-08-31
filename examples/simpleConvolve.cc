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

#include "lsst/utils/Utils.h"
#include "lsst/pex/exceptions.h"
#include "lsst/log/Log.h"
#include "lsst/afw/math.h"
#include "lsst/afw/image.h"

const std::string outImagePath("scOut.fits");
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

int main(int argc, char **argv) {
    typedef afwMath::Kernel::Pixel Pixel;
    unsigned int kernelCols = 6;
    unsigned int kernelRows = 5;

    LOG_CONFIG();
    LOG_SET_LVL("TRACE5.afw.math.convolve", LOG_LVL_INFO);

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
