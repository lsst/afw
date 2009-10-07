// -*- lsst-c++ -*-
#include <iostream>
#include <cmath>
#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Background.h"
#include "lsst/afw/math/Interpolate.h"

using namespace std;
namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

typedef image::Image<float> ImageF;
typedef math::Background Back;

int main() {

    // set the parameters for a fake image.
    int const WID = 256;                // pixels
    int const XCEN = WID/2;             // pixels
    int const YCEN = WID/2;             // pixels
    int const XSIG = 2;                 // pixels
    int const YSIG = XSIG;              // pixels
    float const SKY = 100.0;                 // photo-e
    float const A = 100.0;                   // peak star brightness in photo-e
    int const NUMSTAR = 100;
    
    // declare an image.
    ImageF img(WID, WID);
    img = SKY;
    
    // put sky and some fake stars in the image, and add uniform noise
    for (int i_s = 0; i_s < NUMSTAR; ++i_s) {
        int const xStar = static_cast<int>(WID * static_cast<float>(rand())/RAND_MAX);
        int const yStar = static_cast<int>(WID * static_cast<float>(rand())/RAND_MAX);
        for (int i_y = 0; i_y != img.getHeight(); ++i_y) {
            int i_x = 0;
            for (ImageF::x_iterator ip = img.row_begin(i_y); ip != img.row_end(i_y); ++ip) {

                // use a bivariate gaussian as a stellar PSF
                *ip += A*exp( -((i_x - xStar)*(i_x - xStar) + (i_y - yStar)*(i_y - yStar))/(2.0*XSIG*YSIG) );

                // add the noise on the last pass
                if (i_s == NUMSTAR - 1) {
                    /// \todo Change to a Poisson variate
                    *ip += sqrt(*ip)*2.0*(static_cast<float>(rand())/RAND_MAX - 0.5); 
                }
                ++i_x;
            }
        }
    }

    // declare a background control object for a natural spline
    math::BackgroundControl bgCtrl(math::NATURAL_SPLINE);

    // we can control the background estimate
    bgCtrl.setNxSample(5);
    bgCtrl.setNySample(5);

    // we can also control the statistics
    bgCtrl.sctrl.setNumIter(3);
    bgCtrl.sctrl.setNumSigmaClip(2.5);

    // initialize a background object (derivates for interpolation are computed in the constructor
    Back back = math::makeBackground(img, bgCtrl);
    
    // can get an individual pixel or a whole frame.
    float const MID = back.getPixel(XCEN, YCEN);
    ImageF::Ptr bg = back.getImage<ImageF::Pixel>();
    
    // create a background-subtracted image
    ImageF sub(img.getDimensions());
    sub <<= img;
    sub -= *bg;
    
    // output what we've made
    cout << XCEN << " " << YCEN << " center pixel: " << MID << endl;
    img.writeFits("example_Background_fak.fits");
    bg->writeFits("example_Background_bac.fits");
    sub.writeFits("example_Background_sub.fits");

    return 0;

}
