// -*- lsst-c++ -*-
#include <iostream>
#include <cmath>
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Statistics.h"

using namespace std;
namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

typedef image::Image<float> ImageT;
typedef image::MaskedImage<float> MaskedImageT;
typedef math::Statistics ImgTstat;


/**
 * \file imageStatistics.cc - an example of how to use the Statistics class
 * \author Steve Bickerton
 * \date Jan 8, 2009
 */


/*
 *
 */
template<typename Image>
void printStats(Image &img, math::StatisticsControl const &sctrl) {
    
    // initialize a Statistics object with any stats we might want
    ImgTstat stats = math::makeStatistics(img, math::NPOINT | math::STDEV | math::MEAN | math::VARIANCE |
                                           math::ERRORS | math::MIN | math::MAX | math::VARIANCECLIP |
                                           math::MEANCLIP | math::MEDIAN | math::IQRANGE | math::STDEVCLIP,
                                           sctrl);
    
    // get various stats with getValue() and their errors with getError()
    double const npoint      = stats.getValue(math::NPOINT);
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

    // output
    cout << "N          " << npoint << endl;
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
    cout << endl;
    
}


int main() {

    double const pi = M_PI;
    double const pi_2 = pi/2;

    // declare an image and a masked image
    int const wid = 1024;
    ImageT img(wid, wid);
    MaskedImageT mimg(wid, wid);
    std::vector<float> v(0);
    
    // fill it with some noise (Cauchy noise in this case)
    for (int j = 0; j != img.getHeight(); ++j) {
        
        int k = 0;
        MaskedImageT::x_iterator mip = mimg.row_begin(j);
        for (ImageT::x_iterator ip = img.row_begin(j); ip != img.row_end(j); ++ip, ++mip) {
            double const x_uniform = pi*static_cast<ImageT::Pixel>(std::rand())/RAND_MAX;
            double const x_lorentz = x_uniform; //tan(x_uniform - pi_2);

            *ip = x_lorentz;
            
            // mask the odd rows
            // variance actually diverges for Cauchy noise ... but stats doesn't access this.
            *mip = MaskedImageT::Pixel(x_lorentz, (k%2) ? 0x1 : 0x0, 10.0);

            v.push_back(x_lorentz);
            k++;
        }
    }

    // make a statistics control object and override some of the default properties
    math::StatisticsControl sctrl;
    sctrl.setNumIter(3);
    sctrl.setNumSigmaClip(5.0);
    sctrl.setAndMask(0x1);        // pixels with this mask bit set will be ignored.


    // ==================================================================
    // Get stats for the Image, MaskedImage, and vector
    printStats(img, sctrl);
    printStats(mimg, sctrl);
    printStats(v, sctrl);

    
    return 0;

}
