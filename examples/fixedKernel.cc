// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>

#include "boost/shared_ptr.hpp"
#include "boost/format.hpp"

#include "lsst/afw/image.h"
#include "lsst/afw/math.h"

using namespace std;

int main() {
    typedef lsst::afw::math::Kernel::Pixel Pixel;

    double majorSigma = 2.5;
    double minorSigma = 2.0;
    double angle = 0.5;
    unsigned int kernelCols = 5;
    unsigned int kernelRows = 4;

    lsst::afw::math::GaussianFunction2<Pixel> gaussFunc(majorSigma, minorSigma, angle);
    lsst::afw::math::AnalyticKernel analyticKernel(kernelCols, kernelRows, gaussFunc);
    lsst::afw::image::Image<Pixel> analyticImage(analyticKernel.getDimensions());
    (void)analyticKernel.computeImage(analyticImage, true);
    analyticImage *= 47.3; // denormalize by some arbitrary factor
    
    lsst::afw::math::FixedKernel fixedKernel(analyticImage);

    cout << boost::format("Gaussian kernel with majorSigma=%.1f, minorSigma=%.1f\n") %
        majorSigma % minorSigma;

    lsst::afw::math::printKernel(fixedKernel, true);
}
