#include <iostream>
#include <cmath>

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Statistics.h"

using namespace std;
namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

typedef image::Image<float> ImageT;

int main() {
    ImageT img(10,40);
    img = 100000;

    {
        math::Statistics stats = math::Statistics(img, math::NPOINT | math::MEAN | math::STDEV);
        cout << "Npixel: " << stats.getValue(math::NPOINT) << endl;
        cout << "Mean: " << stats.getValue(math::MEAN) << endl;
        cout << "Error in mean: " << stats.getError(math::MEAN) << " (expect NaN)" << endl;
        cout << "Standard Deviation: " << stats.getValue(math::STDEV) << endl << endl;
    }

    {
        math::Statistics stats = math::Statistics(img, math::STDEV | math::MEAN | math::ERRORS);
        std::pair<double, double> const mean = stats.getResult(math::MEAN);

        cout << "Mean: " << mean.first << " error in mean: " << mean.second << endl << endl;
    }

    {
        math::Statistics stats = math::Statistics(img, math::NPOINT);
        try {
            stats.getValue(math::MEAN);
        } catch (lsst::pex::exceptions::InvalidParameterException &e) {
            cout << "You didn't ask for the mean, so we caught an exception: " << e.what() << endl;
        }
    }

    return 0;
}
