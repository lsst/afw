// -*- lsst-c++ -*-

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
 
/// \file
#include "lsst/afw/image/Image.h"
namespace geom=lsst::afw::geom;
namespace image = lsst::afw::image;
typedef image::Image<int> ImageT;

int main() {
    ImageT in(geom::Extent2I(10, 6));

    // Set data to a ramp
    for (int y = 0; y != in.getHeight(); ++y) {
        for (ImageT::xy_locator ptr = in.xy_at(0, y), end = in.xy_at(in.getWidth(), y);
             ptr != end; ++ptr.x()) {
            *ptr = y;
        }
    }
    //
    // Convolve with a pseudo-Gaussian kernel ((1, 2, 1), (2, 4, 2), (1, 2, 1))
    //
    ImageT out(in.getDimensions()); // Make an output image the same size as the input image
    out.assign(in);
    for (int y = 1; y != in.getHeight() - 1; ++y) {
        for (ImageT::xy_locator ptr =  in.xy_at(1, y), end = in.xy_at(in.getWidth() - 1, y),
                               optr = out.xy_at(1, y); ptr != end; ++ptr.x(), ++optr.x()) {
            *optr = ptr(-1,-1) + 2*ptr(0,-1) +   ptr(1,-1) + 
                  2*ptr(-1, 0) + 4*ptr(0, 0) + 2*ptr(1, 0) + 
                    ptr(-1, 1) + 2*ptr(0, 1) +   ptr(1, 1);
        }
    }
    //
    // Do the same thing a faster way, using cached_location_t
    //
    ImageT::Ptr out2(new ImageT(in.getDimensions()));
    out2->assign(in);

    typedef ImageT::const_xy_locator xy_loc;

    for (int y = 1; y != in.getHeight() - 1; ++y) {
        
        // "dot" means "cursor location" in emacs
        xy_loc dot = in.xy_at(1, y), end = in.xy_at(in.getWidth() - 1, y); 

        xy_loc::cached_location_t nw = dot.cache_location(-1,-1);
        xy_loc::cached_location_t n  = dot.cache_location( 0,-1);
        xy_loc::cached_location_t ne = dot.cache_location( 1,-1);
        xy_loc::cached_location_t w  = dot.cache_location(-1, 0);
        xy_loc::cached_location_t c  = dot.cache_location( 0, 0);
        xy_loc::cached_location_t e  = dot.cache_location( 1, 0);
        xy_loc::cached_location_t sw = dot.cache_location(-1, 1);
        xy_loc::cached_location_t s  = dot.cache_location( 0, 1);
        xy_loc::cached_location_t se = dot.cache_location( 1, 1);

        for (ImageT::x_iterator optr = out2->row_begin(y) + 1; dot != end; ++dot.x(), ++optr) {
            *optr = dot[nw] + 2*dot[n] +   dot[ne] +
                  2*dot[w]  + 4*dot[c] + 2*dot[e] +
                    dot[sw] + 2*dot[s] +   dot[se];
        }
    }
    //
    // Do the same calculation, but set nw etc. outside the loop
    //
    xy_loc pix11 = in.xy_at(1, 1);

    xy_loc::cached_location_t nw = pix11.cache_location(-1,-1);
    xy_loc::cached_location_t n  = pix11.cache_location( 0,-1);
    xy_loc::cached_location_t ne = pix11.cache_location( 1,-1);
    xy_loc::cached_location_t w  = pix11.cache_location(-1, 0);
    xy_loc::cached_location_t c  = pix11.cache_location( 0, 0);
    xy_loc::cached_location_t e  = pix11.cache_location( 1, 0);
    xy_loc::cached_location_t sw = pix11.cache_location(-1, 1);
    xy_loc::cached_location_t s  = pix11.cache_location( 0, 1);
    xy_loc::cached_location_t se = pix11.cache_location( 1, 1);

    for (int y = 1; y != in.getHeight() - 1; ++y) {
        // "dot" means "cursor location" in emacs
        xy_loc dot = in.xy_at(1, y), end = in.xy_at(in.getWidth() - 1, y); 

        for (ImageT::x_iterator optr = out2->row_begin(y) + 1; dot != end; ++dot.x(), ++optr) {
            *optr = dot[nw] + 2*dot[n] +   dot[ne] +
                  2*dot[w]  + 4*dot[c] + 2*dot[e] +
                    dot[sw] + 2*dot[s] +   dot[se];
        }
    }
    //
    // Normalise the kernel.  I.e. divide the smoothed parts of image2 by 16
    //
    {
        ImageT center = ImageT(
            *out2, 
            geom::Box2I(
                geom::Point2I(1, 1), in.getDimensions() - geom::Extent2I(2)
            ), 
            image::LOCAL
        );
        center /= 16;
    }
    //
    // Clear in using the x_iterator embedded in the locator
    //
    for (int y = 0; y != in.getHeight(); ++y) {
        for (ImageT::xy_x_iterator ptr = in.xy_at(0, y).x(), end = in.xy_at(in.getWidth(), y).x();
             ptr != end; ++ptr) {
            *ptr = 0;
        }
    }
    //
    // Save those images to disk
    //
    out.writeFits("foo.fits");
    out2->writeFits("foo2.fits");

    return 0;
}
