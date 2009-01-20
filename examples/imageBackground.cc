// -*- lsst-c++ -*-
#include <iostream>
#include <cmath>
#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Background.h"
#include "lsst/afw/math/Interpolate.h"

using namespace std;
namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

typedef image::Image<float> ImageT;
typedef math::Background BackT;

int main() {

    // set the parameters for a fake image.
    int const wid = 256;                // pixels
    int const xcen = wid/2;             // pixels
    int const ycen = wid/2;             // pixels
    int const xsig = 2;                 // pixels
    int const ysig = xsig;              // pixels
    float const sky = 100.0;                 // photo-e
    float const A = 100.0;                   // peak star brightness in photo-e
    int const numStar = 100;
    
    // declare an image.
    ImageT img(wid, wid);
    img = sky;
    
    // put sky and some fake stars in the image, and add uniform noise
    for (int i_s = 0; i_s < numStar; ++i_s) {
        int const xStar = static_cast<int>(wid * static_cast<float>(rand())/RAND_MAX);
        int const yStar = static_cast<int>(wid * static_cast<float>(rand())/RAND_MAX);
        for (int i_y = 0; i_y != img.getHeight(); ++i_y) {
            int i_x = 0;
            for (ImageT::x_iterator ip = img.row_begin(i_y); ip != img.row_end(i_y); ++ip, ++i_x) {

                // use a bivariate gaussian as a stellar PSF
                *ip += A*exp( -( (i_x - xStar)*(i_x - xStar) + (i_y - yStar)*(i_y - yStar) )/(2.0*xsig*ysig) );

                // add the noise on the last pass
                if (i_s == numStar - 1) {
                    /// \todo Change to a Poisson variate
                    *ip += sqrt(*ip)*2.0*(static_cast<float>(rand())/RAND_MAX - 0.5); 
                }
                
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
    BackT back = math::make_Background(img, bgCtrl);
    
    // can get an individual pixel or a whole frame.
    float const mid = back.getPixel(xcen,ycen);
    ImageT::Ptr bg = back.getImage<ImageT::Pixel>();
    
    // create a background-subtracted image
    ImageT sub(img.getDimensions());
    sub <<= img;
    sub -= *bg;
    
    // output what we've made
    cout << xcen << " " << ycen << " center pixel: " << mid << endl;
    img.writeFits("example_Background_fak.fits");
    bg->writeFits("example_Background_bac.fits");
    sub.writeFits("example_Background_sub.fits");

    return 0;

}
