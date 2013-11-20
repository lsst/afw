// -*- LSST-C++ -*-

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
#include <cmath>
#include "boost/shared_ptr.hpp"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Background.h"
#include "lsst/afw/math/Interpolate.h"

using namespace std;
namespace image = lsst::afw::image;
namespace math = lsst::afw::math;
namespace geom = lsst::afw::geom;

typedef image::Image<float> ImageF;

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
    ImageF img(geom::Extent2I(wid, wid));
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
    math::BackgroundControl bgCtrl(math::Interpolate::NATURAL_SPLINE);

    // could also use a string! (commented-out, but will work)
    //math::BackgroundControl bgCtrl("NATURAL_SPLINE");
    
    // we can control the background estimate
    bgCtrl.setNxSample(5);
    bgCtrl.setNySample(5);

    // we can also control the statistics
    bgCtrl.getStatisticsControl()->setNumIter(3);
    bgCtrl.getStatisticsControl()->setNumSigmaClip(2.5);

    // initialize a background object
    PTR(math::Background) back = math::makeBackground(img, bgCtrl);
    
    // can get an individual pixel or a whole frame.
    float const MID = boost::dynamic_pointer_cast<math::BackgroundMI>(back)->getPixel(xcen, ycen);
    ImageF::Ptr bg = back->getImage<ImageF::Pixel>();
    
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
