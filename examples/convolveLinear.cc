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
#include "lsst/daf/base.h"
#include "lsst/log/Log.h"
#include "lsst/afw/image.h"
#include "lsst/afw/math.h"

namespace afwImage = lsst::afw::image;
namespace afwMath= lsst::afw::math;

const std::string outImagePath("clOut.fits");

int main(int argc, char **argv) {
    LOG_CONFIG();
    LOG_SET_LVL("TRACE5.afw.math.convolve", LOG_LVL_INFO);

    typedef float ImagePixel;
    unsigned int const KernelCols = 19;
    unsigned int const KernelRows = 19;
    double const MinSigma = 1.5;
    double const MaxSigma = 4.5;

    std::string inImagePath;
    if (argc < 2) {
        try {
            std::string dataDir = lsst::utils::getPackageDir("afwdata");
            inImagePath = dataDir + "/data/med.fits";
        } catch (lsst::pex::exceptions::NotFoundError) {
            std::cerr << "Usage: linearConvolve [fitsFile]" << std::endl;
            std::cerr << "fitsFile is the path to a masked image" << std::endl;
            std::cerr << "\nError: setup afwdata or specify fitsFile.\n" << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        inImagePath = std::string(argv[1]);
    }

    // block in which to allocate and deallocate memory
    {
        // read in fits file
        afwImage::MaskedImage<ImagePixel> mImage(inImagePath);

        // construct basis kernels

        afwMath::KernelList kernelList;
        for (int ii = 0; ii < 3; ++ii) {
            double majorSigma = (ii == 1) ? MaxSigma : MinSigma;
            double minorSigma = (ii == 2) ? MinSigma : MaxSigma;
            double angle = 0.0;
            afwMath::GaussianFunction2<afwMath::Kernel::Pixel> gaussFunc(majorSigma, minorSigma, angle);
            std::shared_ptr<afwMath::Kernel> basisKernelPtr(
                new afwMath::AnalyticKernel(KernelCols, KernelRows, gaussFunc)
            );
            kernelList.push_back(basisKernelPtr);
        }

        // construct spatially varying linear combination kernel
        int const polyOrder = 1;
        afwMath::PolynomialFunction2<double> polyFunc(polyOrder);
        afwMath::LinearCombinationKernel kernel(kernelList, polyFunc);

        // Get copy of spatial parameters (all zeros), set and feed back to the kernel
        std::vector<std::vector<double> > polyParams = kernel.getSpatialParameters();
        // Set spatial parameters for basis kernel 0
        polyParams[0][0] =  1.0;
        polyParams[0][1] = -0.5 / static_cast<double>(mImage.getWidth());
        polyParams[0][2] = -0.5 / static_cast<double>(mImage.getHeight());
        // Set spatial function parameters for basis kernel 1
        polyParams[1][0] = 0.0;
        polyParams[1][1] = 1.0 / static_cast<double>(mImage.getWidth());
        polyParams[1][2] = 0.0;
        // Set spatial function parameters for basis kernel 2
        polyParams[2][0] = 0.0;
        polyParams[2][1] = 0.0;
        polyParams[2][2] = 1.0 / static_cast<double>(mImage.getHeight());
        // Set spatial function parameters for kernel parameter 1
        kernel.setSpatialParameters(polyParams);

        std::cerr << "Image: " << inImagePath << std::endl;
        std::cout << "Image size: " << mImage.getWidth() << " x " << mImage.getHeight() << std::endl;
        std::cout << "Kernel size: " << KernelCols << " x " << KernelRows << std::endl;
        std::cout << "Number of basis kernels: " << kernel.getNBasisKernels() << std::endl;
        std::cout << "Spatial order: " << polyOrder << std::endl;

        // convolve
        afwImage::MaskedImage<ImagePixel> resMaskedImage(mImage.getDimensions());
        afwMath::convolve(resMaskedImage, mImage, kernel, false);

        // write results
        resMaskedImage.writeFits(outImagePath);
        std::cout << "Saved convolved image as " << outImagePath << std::endl;
    }

     // Check for memory leaks
     if (lsst::daf::base::Citizen::census(0) != 0) {
         std::cerr << "Leaked memory blocks:" << std::endl;
         lsst::daf::base::Citizen::census(std::cerr);
     }

}
