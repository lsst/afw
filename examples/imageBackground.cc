// -*- LSST-C++ -*-
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
    int const wid = 256;                // pixels
    int const xcen = wid/2;             // pixels
    int const ycen = wid/2;             // pixels
    int const xsig = 2;                 // pixels
    int const ysig = xsig;              // pixels
    float const sky = 100.0;                 // photo-e
    float const A = 100.0;                   // peak star brightness in photo-e
    int const nStar = 100;
    
    // declare an image.
    ImageF img(wid, wid);
    img = sky;
    
    // put sky and some fake stars in the image, and add uniform noise
    for (int iS = 0; iS < nStar; ++iS) {
        int const xStar = static_cast<int>(wid * static_cast<float>(rand())/RAND_MAX);
        int const yStar = static_cast<int>(wid * static_cast<float>(rand())/RAND_MAX);
        for (int i_y = 0; i_y != img.getHeight(); ++i_y) {
            int iX = 0;
            for (ImageF::x_iterator ip = img.row_begin(i_y); ip != img.row_end(i_y); ++ip) {

                // use a bivariate gaussian as a stellar PSF
                *ip += A*exp( -((iX - xStar)*(iX - xStar) + (i_y - yStar)*(i_y - yStar))/(2.0*xsig*ysig) );

                // add the noise on the last pass
                if (iS == nStar - 1) {
                    /// \todo Change to a Poisson variate
                    *ip += sqrt(*ip)*2.0*(static_cast<float>(rand())/RAND_MAX - 0.5); 
                }
                ++iX;
            }
        }
    }

    // declare a background control object for a natural spline
    math::BackgroundControl bgCtrl(math::Interp::NATURAL_SPLINE);

    // we can control the background estimate
    bgCtrl.setNxSample(5);
    bgCtrl.setNySample(5);

    // we can also control the statistics
    bgCtrl.sctrl.setNumIter(3);
    bgCtrl.sctrl.setNumSigmaClip(2.5);

    // initialize a background object (derivates for interpolation are computed in the constructor
    Back back = math::makeBackground(img, bgCtrl);
    
    // can get an individual pixel or a whole frame.
    float const MID = back.getPixel(xcen, ycen);
    ImageF::Ptr bg = back.getImage<ImageF::Pixel>();
    
    // create a background-subtracted image
    ImageF sub(img.getDimensions());
    sub <<= img;
    sub -= *bg;
    
    // output what we've made
    cout << xcen << " " << ycen << " center pixel: " << MID << endl;
    img.writeFits("example_Background_fak.fits");
    bg->writeFits("example_Background_bac.fits");
    sub.writeFits("example_Background_sub.fits");

    return 0;

}
