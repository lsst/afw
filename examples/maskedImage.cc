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

#include <iostream>
#include <string>
#include <algorithm>
#include "boost/iterator/zip_iterator.hpp"
#include "lsst/afw/image/MaskedImage.h"

namespace geom = lsst::afw::geom;
namespace image = lsst::afw::image;
using namespace std;

typedef double ImagePixelT;


template <typename PixelT>
void y_gradient(image::MaskedImage<PixelT> & src, image::MaskedImage<PixelT> & dst) {
    assert(src.getDimensions() == dst.getDimensions());

#define CONST 1
#if CONST
    typedef typename image::MaskedImage<PixelT>::const_xy_locator xyl;
#else
    typedef typename image::MaskedImage<PixelT>::xy_locator xyl;
#endif
    xyl src_loc = src.xy_at(0, 1);

#define USE_CACHE_LOCATION 1
#if USE_CACHE_LOCATION
    typename xyl::cached_location_t above = src_loc.cache_location(0,  1);
    typename xyl::cached_location_t below = src_loc.cache_location(0, -1);
#endif

    for (int r = 1; r < src.getHeight() - 1; ++r) {
        for (typename image::MaskedImage<PixelT>::x_iterator dst_it = dst.row_begin(r);
            dst_it != dst.row_end(r); ++dst_it, ++src_loc.x()) {
#if USE_CACHE_LOCATION                  // this version is faster
            dst_it.image() = (src_loc.image(above) - src_loc.image(below))/2;
#else  // but this is possible too, and more general (but slower)
            dst_it.image() = (src_loc.image(0, 1) - src_loc.image(0, -1))/2;
#endif
            dst_it.mask()    |= src_loc.mask(0, 1)     | src_loc.mask(0, -1);
            dst_it.variance() = src_loc.variance(0, 1) + src_loc.variance(0, -1);

            //src_loc.image()++;            // uncomment to check const checking
            //src_loc.mask() |= 2;          // uncomment to check const checking
            //src_loc.variance() = 0.0;     // uncomment to check const checking
        }

        src_loc += std::make_pair(-src.getWidth(), 1);
    }
}


namespace {
    void printImage(image::MaskedImage<ImagePixelT> const& img, string const& title="") {
        if (title != "") {
            cout << title << endl;
        }

        for (int i = img.getHeight() - 1; i >= 0; --i) {
            for (image::MaskedImage<ImagePixelT>::x_iterator ptr = img.row_begin(i), end = img.row_end(i);
                 ptr != end; ++ptr) {
                cout << ptr.image() << " ";
            }
            cout << endl;
        }
    }

    void printVariance(image::MaskedImage<ImagePixelT> const& img, string const& title="") {
        if (title != "") {
            cout << title << endl;
        }

        for (int i = img.getHeight() - 1; i >= 0; --i) {
            for (image::MaskedImage<ImagePixelT>::x_iterator ptr = img.row_begin(i), end = img.row_end(i);
                 ptr != end; ++ptr) {
                cout << ptr.variance() << " ";
            }
            cout << endl;
        }
    }
}


int main() {
    image::MaskedImage<ImagePixelT> img(geom::Extent2I(3, 5));
    *img.getImage() = 100;
    *img.getMask() = 0x10;
    *img.getVariance() = 10;

    {
        float const gain = 2;
        img.getVariance()->assign(image::Image<image::VariancePixel>(*img.getImage(), true));
        *img.getVariance() /= gain;
    }

    int i = 0;
    for (image::MaskedImage<ImagePixelT>::iterator ptr = img.begin(), end = img.end();
         ptr != end; ++ptr, ++i) {
        ptr.image() = i/img.getWidth() + 100*(i%img.getWidth());
        ptr.mask() |= 0x8;
        ptr.variance() *= 2;
    }

#if 1
    for (image::MaskedImage<ImagePixelT>::const_iterator ptr = img.at(0,2), end = img.end();
         ptr != end; ++ptr) {
        cout << ptr.image() << " " << ptr.mask() << " " << ptr.variance() << endl;
    }
    cout << endl;

    for (image::MaskedImage<ImagePixelT>::x_iterator ptr = img.row_begin(0), end = img.row_end(0);
         ptr != end; ++ptr) {
        cout << ptr.image() << " " << ptr.mask() << " " << ptr.variance() << endl;
    }
    cout << endl;

    for (image::MaskedImage<ImagePixelT>::reverse_iterator ptr = img.rbegin(), rend = img.rend();
         ptr != rend; ++ptr) {
        cout << ptr.image() << " " << ptr.mask() << " " << ptr.variance() << endl;
    }

    for (image::MaskedImage<ImagePixelT>::y_iterator ptr = img.col_begin(1), end = img.col_end(1);
         ptr != end; ++ptr) {
        cout << ptr.image() << " " << ptr.mask() << " " << ptr.variance() << endl;
    }
    cout << endl;

    for (image::MaskedImage<ImagePixelT>::const_x_iterator ptr = img.x_at(1,1), end = img.x_at(5,1);
         ptr != end; ++ptr) {
        cout << ptr.image() << " " << ptr.mask() << " " << ptr.variance() << endl;
    }
    cout << endl;

#endif
    for (image::MaskedImage<ImagePixelT>::y_iterator ptr = img.y_at(1,0), end = img.y_at(1,2);
         ptr != end; ++ptr) {
        cout << ptr.image() << " " << ptr.mask() << " " << ptr.variance() << endl;
    }
    cout << endl;

#if 1
    image::MaskedImage<ImagePixelT> grad_y(img.getDimensions());
    *grad_y.getImage() = 0;
    y_gradient(img, grad_y);

    printImage(img, "image");
    cout << endl;
    printVariance(img, "variance(image)");
    cout << endl;

    printImage(grad_y, "gradient");
    cout << endl;
    printVariance(grad_y, "variance(gradient)");
    cout << endl;
#endif

    return 0;
}
