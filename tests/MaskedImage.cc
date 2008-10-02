#include <iostream>
#include <string>
#include <algorithm>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE MaskedImage

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "boost/iterator/zip_iterator.hpp"
#include "lsst/afw/image/MaskedImage.h"

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

    void printMask(image::MaskedImage<float> const& img, string const& title="") {
        if (title != "") {
            cout << title << endl;
        }
        
        for (int i = img.getHeight() - 1; i >= 0; --i) {
            for (image::MaskedImage<float>::x_iterator ptr = img.row_begin(i), end = img.row_end(i); ptr != end; ++ptr) {
                cout << ptr.mask() << " ";
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

image::MaskedImage<float> make_image(int const width=5, int const height=6) {
    image::MaskedImage<float> img(width, height);

    int i = 0;
    for (image::MaskedImage<float>::iterator ptr = img.begin(), end = img.end(); ptr != end; ++ptr, ++i) {
        ptr.image() = i/img.getWidth() + 100*(i%img.getWidth());
        ptr.mask() = i;;
        ptr.variance() = 2*ptr.image();
    }

    return img;
}

BOOST_AUTO_TEST_CASE(setValues) {
    image::MaskedImage<float> img = make_image();

#if 0
    printImage(img, "Image");
    printMask(img, "Mask");
    printVariance(img, "Variance");
#endif

    BOOST_CHECK_EQUAL((*img.getImage())(1,1), 101);
    BOOST_CHECK_EQUAL((*img.getMask())(1,1), img.getWidth() + 1);
    BOOST_CHECK_EQUAL((*img.getVariance())(1,1), 202);

    image::MaskedImage<float>::x_iterator ptr = img.x_at(1, 1);

    BOOST_CHECK_EQUAL(ptr.image(),    101);
    BOOST_CHECK_EQUAL(ptr.mask(),     img.getWidth() + 1);
    BOOST_CHECK_EQUAL(ptr.variance(), 202);

    *ptr /= 2;
    BOOST_CHECK_EQUAL(ptr.image(),    50.5);
    BOOST_CHECK_EQUAL(ptr.mask(),     img.getWidth() + 1);
    BOOST_CHECK_EQUAL(ptr.variance(), 50.5);

    *ptr *= 2;
    BOOST_CHECK_EQUAL(ptr.image(),    101);
    BOOST_CHECK_EQUAL(ptr.mask(),     img.getWidth() + 1);
    BOOST_CHECK_EQUAL(ptr.variance(), 202);

    *ptr /= *ptr;
    BOOST_CHECK_EQUAL(ptr.image(),    1);
    BOOST_CHECK_EQUAL(ptr.mask(),     img.getWidth() + 1);
    BOOST_CHECK_CLOSE(ptr.variance(), 2*202/float(101*101), 1e-8); // 3rd argument is allowed percentage difference

    ptr++;
    BOOST_CHECK_EQUAL(ptr.image(),    201);
    BOOST_CHECK_EQUAL(ptr.mask(),     img.getWidth() + 2);
    BOOST_CHECK_EQUAL(ptr.variance(), 402);

    image::MaskedImage<float> img2(img.dimensions()); // make a deep copy
    *img2.getImage() = 4;
    *img2.getMask() = 0x8;
    *img2.getVariance() = 8;

    image::MaskedImage<float>::x_iterator ptr2 = img2.x_at(0, 4);

    ptr.image() = 100;
    ptr.mask() = 0x10;
    ptr.variance() = 3;
    BOOST_CHECK_EQUAL(ptr.image(),    100);
    BOOST_CHECK_EQUAL(ptr.mask(),     0x10);
    BOOST_CHECK_EQUAL(ptr.variance(), 3);

    *ptr *= *img2.x_at(0,4);            // == ptr2
    BOOST_CHECK_EQUAL(ptr.image(),    400);
    BOOST_CHECK_EQUAL(ptr.mask(),     0x18);
    BOOST_CHECK_EQUAL(ptr.variance(), 100*100*8 + 4*4*3);

    *ptr2 += 10;
    BOOST_CHECK_EQUAL(ptr2.image(),    14);
    BOOST_CHECK_EQUAL(ptr2.mask(),     0x8);
    BOOST_CHECK_EQUAL(ptr2.variance(), 8);

    *ptr2 -= 10;
    BOOST_CHECK_EQUAL(ptr2.image(),    4);
    BOOST_CHECK_EQUAL(ptr2.mask(),     0x8);
    BOOST_CHECK_EQUAL(ptr2.variance(), 8);

    *img.getImage() = 10;
    *img.getMask() = 0x1;
    *img.getVariance() = 2.5;
    
    *ptr += *ptr2;
    BOOST_CHECK_EQUAL(ptr.image(),    14);
    BOOST_CHECK_EQUAL(ptr.mask(),     0x9);
    BOOST_CHECK_EQUAL(ptr.variance(), 10.5);    
    
    ptr2.mask() = 0x2;
    *ptr -= *ptr2;
    BOOST_CHECK_EQUAL(ptr.image(),    10);
    BOOST_CHECK_EQUAL(ptr.mask(),     0x9 | 0x2);
    BOOST_CHECK_EQUAL(ptr.variance(), 18.5);
}

/************************************************************************************************************/
//
// Iterators
//
BOOST_AUTO_TEST_CASE(iterators) {
    image::MaskedImage<float> img = make_image();
    //
    // Count the pixels between begin() and end() (using a const_iterator)
    //
    {
        int i = 0;
        for (image::MaskedImage<float>::const_iterator ptr = img.begin(), end = img.end(); ptr != end; ++ptr, ++i) {
        }
        BOOST_CHECK_EQUAL(i, img.getWidth()*img.getHeight());
    }
    //
    // Count the pixels between begin() and end() backwards (there is no const_reverse_iterator)
    //
    {
        int i = 0;
        for (image::MaskedImage<float>::reverse_iterator ptr = img.rbegin(), end = img.rend(); ptr != end; ++ptr, ++i) {
        }
        BOOST_CHECK_EQUAL(i, img.getWidth()*img.getHeight());
    }
    //
    // Check begin() and our ability to increment it
    //        
    {
        image::MaskedImage<float>::iterator ptr = img.begin();
        ptr += img.getWidth() + 1;           // move to (1,1)
        
        BOOST_CHECK_EQUAL(ptr.image(), 101);
        BOOST_CHECK_EQUAL(ptr.mask(), img.getWidth() + 1);
        BOOST_CHECK_EQUAL(ptr.variance(), 202);
    }
    //
    // Check {col,row}_begin() and our ability to increment them
    //        
    {
        image::MaskedImage<float>::x_iterator rptr = img.row_begin(1);
        rptr += 1;                       // move to (1,1)
        
        BOOST_CHECK_EQUAL(rptr.image(), 101);
        BOOST_CHECK_EQUAL(rptr.mask(), img.getWidth() + 1);
        BOOST_CHECK_EQUAL(rptr.variance(), 202);

        BOOST_REQUIRE(img.getWidth() >= 4);
        image::MaskedImage<float>::const_x_iterator at_ptr = img.x_at(3,2);
        BOOST_CHECK_EQUAL(at_ptr.image(), 302);
    }
    {
        image::MaskedImage<float>::y_iterator cptr = img.col_begin(2);
        cptr += 1;                       // move to (2,1)
        
        BOOST_CHECK_EQUAL(cptr.image(), 201);
        BOOST_CHECK_EQUAL(cptr.mask(), img.getWidth() + 2);
        BOOST_CHECK_EQUAL(cptr.variance(), 402);

        BOOST_REQUIRE(img.getWidth() >= 4);
        image::MaskedImage<float>::const_y_iterator at_ptr = img.y_at(3,2);
        BOOST_CHECK_EQUAL(at_ptr.image(), 302);
    }
    //
    // Test {col,row}_{begin,end} by finding the width and height
    //
    {
        int i = 0;
        for (image::MaskedImage<float>::x_iterator ptr = img.row_begin(0), end = img.row_end(0); ptr != end; ++ptr) {
            ++i;
        }
        BOOST_CHECK_EQUAL(i, img.getWidth());

        i = 0;
        for (image::MaskedImage<float>::y_iterator ptr = img.col_begin(0), end = img.col_end(0); ptr != end; ++ptr) {
            ++i;
        }
        BOOST_CHECK_EQUAL(i, img.getHeight());
    }
    //
    // Check that at() works too
    //
    {
        image::MaskedImage<float>::const_iterator ptr = img.at(1,1);
        
        BOOST_CHECK_EQUAL(ptr.image(), 101);
        BOOST_CHECK_EQUAL(ptr.mask(), img.getWidth() + 1);
        BOOST_CHECK_EQUAL(ptr.variance(), 202);
    }
}

/************************************************************************************************************/
//
// Locators
//
BOOST_AUTO_TEST_CASE(locators) {
    image::MaskedImage<float> img = make_image();

    {
        image::MaskedImage<float>::const_xy_locator loc = img.xy_at(1,1);

        BOOST_CHECK_EQUAL(loc.image(), 101);
        BOOST_CHECK_EQUAL(loc.mask(), img.getWidth() + 1);
        BOOST_CHECK_EQUAL(loc.variance(), 202);

        BOOST_CHECK_EQUAL(loc.image(1,1), 202);
        BOOST_CHECK_EQUAL(loc.mask(1,1), 2*img.getWidth() + 2);
        BOOST_CHECK_EQUAL(loc.variance(1,1), 404);

        loc += std::make_pair(-1, 1); // loc == img.xy_at(0, 2);
        BOOST_CHECK_EQUAL(loc.image(), 2);
        BOOST_CHECK_EQUAL(loc.mask(), 2*img.getWidth());
        BOOST_CHECK_EQUAL(loc.variance(), 4);

        loc.x() += 2; ++loc.x();        // loc == img.xy_at(3, 2);
        loc.y() += 1; loc.y() += 1;     // loc == img.xy_at(3, 4);
        BOOST_REQUIRE(img.getWidth() >= 4);
        BOOST_REQUIRE(img.getHeight() >= 5);
        
        BOOST_CHECK_EQUAL(loc.image(), 304);
        BOOST_CHECK_EQUAL(loc.mask(), 4*img.getWidth() + 3);
        BOOST_CHECK_EQUAL(loc.variance(), 608);

        BOOST_CHECK_EQUAL(loc.x().image(), 304);
        BOOST_CHECK_EQUAL(loc.x().mask(), 4*img.getWidth() + 3);
        BOOST_CHECK_EQUAL(loc.x().variance(), 608);

        BOOST_CHECK_EQUAL(loc.y().image(), 304);
        BOOST_CHECK_EQUAL(loc.y().mask(), 4*img.getWidth() + 3);
        BOOST_CHECK_EQUAL(loc.y().variance(), 608);

        ++loc.x();
        BOOST_CHECK_EQUAL(loc.x().image(), 404);
        BOOST_CHECK_EQUAL(loc.image(), 404);
    }

    {
        image::MaskedImage<float>::xy_locator loc = img.xy_at(1,1);
        image::MaskedImage<float>::xy_locator::cached_location_t above = loc.cache_location(0,  1);
        image::MaskedImage<float>::xy_locator::cached_location_t below = loc.cache_location(0, -1);
        
        BOOST_CHECK_EQUAL(loc.image(above), 102);
        BOOST_CHECK_EQUAL(loc.mask(above), 2*img.getWidth() + 1);
        BOOST_CHECK_EQUAL(loc.variance(below), 200);
    }    
}
