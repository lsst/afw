// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>
#include <cmath>

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Statistics.h"

using namespace std;
namespace image = lsst::afw::image;
namespace math = lsst::afw::math;
namespace geom = lsst::afw::geom;

typedef image::Image<float> ImageF;

int main() {

    // First we'll try a regular image
    ImageF img(geom::Extent2I(10, 40));
    img = 100000.0;
    
    {
        math::Statistics stats = math::makeStatistics(img, math::NPOINT | math::MEAN | math::STDEV);
        cout << "Npixel: " << stats.getValue(math::NPOINT) << endl;
        cout << "Mean: " << stats.getValue(math::MEAN) << endl;
        cout << "Error in mean: " << stats.getError(math::MEAN) << " (expect NaN)" << endl;
        cout << "Standard Deviation: " << stats.getValue(math::STDEV) << endl << endl;
    }

    {
        math::Statistics stats = math::makeStatistics(img, math::STDEV | math::MEAN | math::ERRORS);
        std::pair<double, double> mean = stats.getResult(math::MEAN);

        cout << "Mean: " << mean.first << " error in mean: " << mean.second << endl << endl;
    }

    {
        math::Statistics stats = math::makeStatistics(img, math::NPOINT);
        try {
            stats.getValue(math::MEAN);
        } catch (lsst::pex::exceptions::InvalidParameterError &e) {
            cout << "You didn't ask for the mean, so we caught an exception: " << e.what() << endl;
        }
    }

    return 0;
}
