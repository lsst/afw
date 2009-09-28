#include <iostream>

#include "boost/shared_ptr.hpp"
#include "boost/format.hpp"

#include "lsst/afw/image.h"
#include "lsst/afw/math.h"

using namespace std;

int main() {
    typedef lsst::afw::math::Kernel::Pixel Pixel;

    double sigmaX = 2.0;
    double sigmaY = 2.5;
    unsigned int kernelCols = 5;
    unsigned int kernelRows = 4;

    lsst::afw::math::GaussianFunction2<Pixel> gaussFunc(sigmaX, sigmaY);
    lsst::afw::math::AnalyticKernel analyticKernel(kernelCols, kernelRows, gaussFunc);
    lsst::afw::image::Image<Pixel> analyticImage(analyticKernel.getDimensions());
    (void)analyticKernel.computeImage(analyticImage, true);
    analyticImage *= 47.3; // denormalize by some arbitrary factor
    
    lsst::afw::math::FixedKernel fixedKernel(analyticImage);

    cout << boost::format("Gaussian kernel with sigmaX=%.1f, sigmaY=%.1f\n") % sigmaX % sigmaY;

    lsst::afw::math::printKernel(fixedKernel, true);
}
