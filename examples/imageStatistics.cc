// -*- lsst-c++ -*-
#include <iostream>
#include <cmath>
#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Statistics.h"

using namespace std;
namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

typedef image::Image<float> ImageT;
typedef math::Statistics<ImageT> ImgTstat;


int main() {

    double const pi = 3.14159265358979;
    double const pi_2 = (pi / 2);

    // declare an image
    int const wid = 1024;
    ImageT img(wid, wid);

    // fill it with noise (Cauchy noise in this case)
    for (int j = 0; j != img.getHeight(); ++j) {
        int i = 0;
        for (ImageT::x_iterator ip = img.row_begin(j); ip != img.row_end(j); ++ip, ++i) {
            double x_uniform = pi * static_cast<ImageT::Pixel>(std::rand()) / RAND_MAX;
            double x_lorentz = tan(x_uniform - pi_2);
            *ip = x_lorentz;
        }
    }

    // make a statistics control object and override some properties
    math::StatisticsControl sctrl;
    sctrl.setNumIter(3);
    sctrl.setNumSigmaClip(5.0);

    // initialize a Statistics object
    ImgTstat stats = math::make_Statistics(img, math::NPOINT | math::STDEV | math::MEAN | math::VARIANCE |
                                           math::ERRORS | math::MIN | math::MAX | math::VARIANCECLIP |
                                           math::MEANCLIP | math::MEDIAN | math::IQRANGE | math::STDEVCLIP,
                                           sctrl);

    // get various stats with getValue() and their errors with getError()
    double const mean      = stats.getValue(math::MEAN);
    double const var       = stats.getValue(math::VARIANCE);
    double const dmean     = stats.getError(math::MEAN);
    double const sd        = stats.getValue(math::STDEV);
    double const min       = stats.getValue(math::MIN);
    double const max       = stats.getValue(math::MAX);
    double const meanclip  = stats.getValue(math::MEANCLIP);
    double const varclip   = stats.getValue(math::VARIANCECLIP);
    double const stdevclip = stats.getValue(math::STDEVCLIP);
    double const median    = stats.getValue(math::MEDIAN);
    double const iqrange   = stats.getValue(math::IQRANGE);

    // print to stdout.
    cout << "dmean      " << dmean << endl;

    cout << "mean:      " << mean << endl;
    cout << "meanclip:  " << meanclip << endl;

    cout << "var:       " << var << endl;
    cout << "varclip:   " << varclip << endl;

    cout << "stdev:     " << sd << endl;
    cout << "stdevclip: " << stdevclip << endl;

    cout << "min:       " << min << endl;
    cout << "max:       " << max <<  endl;
    cout << "median:    " << median << endl;
    cout << "iqrange:   " << iqrange << endl;

    return 0;

}
