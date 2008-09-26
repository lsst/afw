#include <iostream>
#include <string>
#include <algorithm>
#include "boost/iterator/zip_iterator.hpp"
#include "lsst/gil/MaskedImage.h"

namespace image = lsst::afw::image;
using namespace std;

/************************************************************************************************************/

template <typename PixelT>
void y_gradient(image::MaskedImage<PixelT> & src, image::MaskedImage<PixelT> & dst) {
    assert(src.dimensions() == dst.dimensions());

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

/************************************************************************************************************/

namespace {
    void printImage(image::MaskedImage<float> const& img, string const& title="") {
        if (title != "") {
            cout << title << endl;
        }
        
        for (int i = img.getHeight() - 1; i >= 0; --i) {
            for (image::MaskedImage<float>::x_iterator ptr = img.row_begin(i), end = img.row_end(i); ptr != end; ++ptr) {
                cout << ptr.image() << " ";
            }
            cout << endl;
        }
    }

    void printVariance(image::MaskedImage<float> const& img, string const& title="") {
        if (title != "") {
            cout << title << endl;
        }
        
        for (int i = img.getHeight() - 1; i >= 0; --i) {
            for (image::MaskedImage<float>::x_iterator ptr = img.row_begin(i), end = img.row_end(i); ptr != end; ++ptr) {
                cout << ptr.variance() << " ";
            }
            cout << endl;
        }
    }
}

/************************************************************************************************************/

int main() {
    image::MaskedImage<float> img(3, 5);
    *img.getImage() = 100;
    *img.getMask() = 0x10;
    *img.getVariance() = 10;

    int i = 0;
    for (image::MaskedImage<float>::iterator ptr = img.begin(), end = img.end(); ptr != end; ++ptr, ++i) {
        ptr.image() = i/img.getWidth() + 100*(i%img.getWidth());
        ptr.mask() |= 0x8;
        ptr.variance() *= 2;
    }

#if 1
    for (image::MaskedImage<float>::const_iterator ptr = img.at(0,2), end = img.end(); ptr != end; ++ptr) {
        cout << ptr.image() << " " << ptr.mask() << " " << ptr.variance() << endl;
    }
    cout << endl;

    for (image::MaskedImage<float>::x_iterator ptr = img.row_begin(0), end = img.row_end(0); ptr != end; ++ptr) {
        cout << ptr.image() << " " << ptr.mask() << " " << ptr.variance() << endl;
    }
    cout << endl;

    for (image::MaskedImage<float>::reverse_iterator ptr = img.rbegin(), rend = img.rend(); ptr != rend; ++ptr) {
        cout << ptr.image() << " " << ptr.mask() << " " << ptr.variance() << endl;
    }

    for (image::MaskedImage<float>::y_iterator ptr = img.col_begin(1), end = img.col_end(1); ptr != end; ++ptr) {
        cout << ptr.image() << " " << ptr.mask() << " " << ptr.variance() << endl;
    }
    cout << endl;

    for (image::MaskedImage<float>::const_x_iterator ptr = img.x_at(1,1), end = img.x_at(5,1); ptr != end; ++ptr) {
        cout << ptr.image() << " " << ptr.mask() << " " << ptr.variance() << endl;
    }
    cout << endl;

#endif
    for (image::MaskedImage<float>::y_iterator ptr = img.y_at(1,0), end = img.y_at(1,2); ptr != end; ++ptr) {
        cout << ptr.image() << " " << ptr.mask() << " " << ptr.variance() << endl;
    }
    cout << endl;

#if 1
    image::MaskedImage<float> grad_y(img.dimensions());
    *grad_y.getImage() = 0;
    y_gradient(img, grad_y);

    printImage(img, "image"); cout << endl;
    printVariance(img, "variance(image)"); cout << endl;
    
    printImage(grad_y, "gradient"); cout << endl;
    printVariance(grad_y, "variance(gradient)");
    cout << endl;
#endif

    return 0;
}
