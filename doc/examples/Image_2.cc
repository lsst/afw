/// \file
#include "lsst/afw/image/Image.h"

namespace image = lsst::afw::image;
typedef image::Image<int> ImageT;

int main() {
    ImageT in(10, 6);
    
    // Set data to a ramp
    for (int y = 0; y != in.getHeight(); ++y) {
        for (ImageT::xy_locator ptr = in.xy_at(0, y),
                                              end = in.xy_at(in.getWidth(), y); ptr != end; ++ptr.x()) {
            *ptr = y;
        }
    }
    //
    // Convolve with a pseudo-Gaussian kernel ((1, 2, 1), (2, 4, 2), (1, 2, 1))
    //
    ImageT out(in.dimensions()); // Make an output image the same size as the input image
    out <<= in;
    for (int y = 1; y != in.getHeight() - 1; ++y) {
        for (ImageT::xy_locator ptr =   in.xy_at(1, y), end = in.xy_at(in.getWidth() - 1, y),
                                              optr = out.xy_at(1, y); ptr != end; ++ptr.x(), ++optr.x()) {
            *optr = ptr(-1,-1) + 2*ptr(0,-1) +   ptr(1,-1) + 
                  2*ptr(-1, 0) + 4*ptr(0, 0) + 2*ptr(1, 0) + 
                    ptr(-1, 1) + 2*ptr(0, 1) +   ptr(1, 1);
        }
    }
    //
    // Do the same thing a faster way, using cached_location_t
    //
    ImageT::Ptr out2(new ImageT(in.dimensions()));
    *out2 <<= in;

    typedef ImageT::const_xy_locator xy_loc;

    for (int y = 1; y != in.getHeight() - 1; ++y) {
        xy_loc dot = in.xy_at(1, y), end = in.xy_at(in.getWidth() - 1, y); // "dot" means "cursor location" in emacs

        xy_loc::cached_location_t nw = dot.cache_location(-1,-1);
        xy_loc::cached_location_t n  = dot.cache_location( 0,-1);
        xy_loc::cached_location_t ne = dot.cache_location( 1,-1);
        xy_loc::cached_location_t w  = dot.cache_location(-1, 0);
        xy_loc::cached_location_t e  = dot.cache_location( 1, 0);
        xy_loc::cached_location_t sw = dot.cache_location(-1, 1);
        xy_loc::cached_location_t s  = dot.cache_location( 0, 1);
        xy_loc::cached_location_t se = dot.cache_location( 1, 1);

        for (ImageT::x_iterator optr = out2->row_begin(y) + 1; dot != end; ++dot.x(), ++optr) {
            *optr = dot[nw] + 2*dot[n] + dot[ne] + 2*dot[w] + 4*(*dot) + 2*dot[e] + dot[sw] + 2*dot[s] + dot[se];
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
    xy_loc::cached_location_t e  = pix11.cache_location( 1, 0);
    xy_loc::cached_location_t sw = pix11.cache_location(-1, 1);
    xy_loc::cached_location_t s  = pix11.cache_location( 0, 1);
    xy_loc::cached_location_t se = pix11.cache_location( 1, 1);

    for (int y = 1; y != in.getHeight() - 1; ++y) {
        xy_loc dot = in.xy_at(1, y), end = in.xy_at(in.getWidth() - 1, y); // "dot" means "cursor location" in emacs

        for (ImageT::x_iterator optr = out2->row_begin(y) + 1; dot != end; ++dot.x(), ++optr) {
            *optr = dot[nw] + 2*dot[n] + dot[ne] + 2*dot[w] + 4*(*dot) + 2*dot[e] + dot[sw] + 2*dot[s] + dot[se];
        }
    }
    //
    // Normalise the kernel.  I.e. divide the smoothed parts of image2 by 16
    //
    {
        ImageT center = ImageT(*out2, image::BBox(image::PointI(1, 1), in.getWidth() - 2, in.getHeight() - 2));
        center /= 16;
    }
    //
    // Save those images to disk
    //
    out.writeFits("foo.fits");
    out2->writeFits("foo2.fits");

    return 0;
}
