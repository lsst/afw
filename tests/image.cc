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

//  -*- lsst-c++ -*-
#include <iostream>
#include <string>
#include <algorithm>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Image

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/MaskedImage.h"

namespace image = lsst::afw::image;
namespace geom = lsst::afw::geom;

using namespace std;

typedef float PixelT;
typedef unsigned short MaskPixelT;
typedef image::Image<PixelT> ImageT;
typedef image::Mask<MaskPixelT> MaskT;


template <typename PixelT>
void y_gradient(ImageT & src, ImageT & dst) {
    assert(src.getDimensions() == dst.getDimensions());

#define CONST 0
#if CONST
    typedef typename ImageT::const_xy_locator xyl;
#else
    typedef typename ImageT::xy_locator xyl;
#endif
    xyl src_loc = src.xy_at(0, 1);

#define USE_CACHE_LOCATION 1
#if USE_CACHE_LOCATION
    typename xyl::cached_location_t above = src_loc.cache_location(0,  1);
    typename xyl::cached_location_t below = src_loc.cache_location(0, -1);
#endif

    for (int r = 1; r < src.getHeight() - 1; ++r) {
        for (typename ImageT::x_iterator dst_it = dst.row_begin(r);
            dst_it != dst.row_end(r); ++dst_it, ++src_loc.x()) {
#if USE_CACHE_LOCATION                  // this version is faster
            (*dst_it) = (src_loc[above] - src_loc[below])/2;
#else  // but this is possible too, and more general (but slower)
            *dst_it = (src_loc(0, 1) - src_loc(0, -1))/2;
#endif
        }

        src_loc += std::make_pair(-src.getWidth(), 1);
    }
}


#define PRINT_IMAGE 0
#if PRINT_IMAGE
namespace {
    void printImage(ImageT const& img, string const& title="") {
        if (title != "") {
            cout << title << endl;
        }

        for (int i = img.getHeight() - 1; i >= 0; --i) {
            for (ImageT::x_iterator ptr = img.row_begin(i), end = img.row_end(i); ptr != end; ++ptr) {
                cout << *ptr << " ";
            }
            cout << endl;
        }
    }
}
#endif

ImageT make_image(int const width=5, int const height=6) {
    ImageT img(geom::Extent2I(width, height));

    int i = 0;
    for (ImageT::iterator ptr = img.begin(), end = img.end(); ptr != end; ++ptr, ++i) {
        *ptr = i/img.getWidth() + 100*(i%img.getWidth());
    }

    return img;
}


BOOST_AUTO_TEST_CASE(getset0) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    ImageT img = make_image();

	image::CheckIndices docheck(true);
	// operator() as getter
	BOOST_CHECK_EQUAL(img(3,3), 303);
	BOOST_CHECK_EQUAL(*(img.x_at(3,3)), 303);
	BOOST_CHECK_EQUAL(img(3,3, docheck), 303);
	BOOST_CHECK_EQUAL(img(3,4), 304);
	BOOST_CHECK_EQUAL(*(img.x_at(3,4)), 304);

	// operator() as setter
	img(3,4) = 3004;
	BOOST_CHECK_EQUAL(img(3,4), 3004);

	// get0 / set0 on a (0,0)-referenced image
	img.set0(3,4, 3004);
	BOOST_CHECK_EQUAL(img(3,4), 3004);
	BOOST_CHECK_EQUAL(img.get0(3,4,docheck), 3004);
	BOOST_CHECK_EQUAL(img.get0(3,4), 3004);
	img.set0(3, 4, 304, docheck);
	BOOST_CHECK_EQUAL(img(3,4), 304);
	img.set0(3, 4, 3004, docheck);
	BOOST_CHECK_EQUAL(img(3,4), 3004);
	img.set0(3, 4, 304);
	BOOST_CHECK_EQUAL(img(3,4), 304);

	// create a subimage:   x in [1,3], y in [2,4]
	ImageT subimg(img, geom::Box2I(geom::Point2I(1,2), geom::Extent2I(3,3)));
	BOOST_CHECK_EQUAL(subimg.getWidth(), 3);
	BOOST_CHECK_EQUAL(subimg.getHeight(), 3);
	BOOST_CHECK_EQUAL(subimg.getX0(), 1);
	BOOST_CHECK_EQUAL(subimg.getY0(), 2);

	BOOST_CHECK_EQUAL(subimg(0,0), 102);
	BOOST_CHECK_EQUAL(subimg(2,2), 304);
	BOOST_CHECK_EQUAL(subimg.get0(1, 2, docheck), 102);
	// (0,0) in the subimage is (1,2) in the parent image
	BOOST_CHECK_EQUAL(subimg.get0(1, 2), 102);
	// subimage can't reference parent coord 0,0
	BOOST_CHECK_THROW(subimg.get0(0, 0, docheck), lsst::pex::exceptions::LengthError);

	subimg.set0(3, 4, 1000000, docheck);
	BOOST_CHECK_EQUAL(subimg.get0(3, 4, docheck), 1000000);
	BOOST_CHECK_EQUAL(subimg(2,2), 1000000);
	BOOST_CHECK_THROW(subimg.set0(0, 0, 1000000, docheck), lsst::pex::exceptions::LengthError);
}


BOOST_AUTO_TEST_CASE(setValues) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    ImageT img = make_image();
    MaskT mask(geom::Extent2I(1, 1));
    mask = 0x8;

#if PRINT_IMAGE
    printImage(img, "Image");
#endif

    ImageT::x_iterator ptr = img.x_at(1, 1);
    MaskT::y_iterator mptr = mask.y_at(0, 0);

    BOOST_CHECK_EQUAL(*ptr,    101);
    BOOST_CHECK_EQUAL(*mptr,   0x8);

    *ptr /= 2;
    *mptr /= 2;
    BOOST_CHECK_EQUAL(*ptr,    50.5);
    BOOST_CHECK_EQUAL(*mptr,   0x4);

    *ptr *= 2;
    BOOST_CHECK_EQUAL(*ptr,    101);

    *ptr /= *ptr;
    BOOST_CHECK_EQUAL(*ptr,    1);

    ptr++;
    BOOST_CHECK_EQUAL(*ptr,    201);

    ImageT img2(img.getDimensions()); // make a deep copy
    img2 = 4;

    ImageT::x_iterator ptr2 = img2.x_at(0, 4);
    //ImageT::Pixel val2 = *ptr2;

    *ptr = 100;                         // sets *ptr to (100, 0, 0)

    BOOST_CHECK_EQUAL(*ptr,    100);

    *ptr *= *img2.x_at(0,4);            // == ptr2
    BOOST_CHECK_EQUAL(*ptr,    400);

    *ptr2 += 10;
    BOOST_CHECK_EQUAL(*ptr2,    14);

    *ptr2 -= 10;
    BOOST_CHECK_EQUAL(*ptr2,    4);

    img = 10;

    *ptr += *ptr2;
    BOOST_CHECK_EQUAL(*ptr,    14);

    *ptr -= *ptr2;
    BOOST_CHECK_EQUAL(*ptr,    10);

    *ptr = *ptr + *ptr2;
    BOOST_CHECK_EQUAL(*ptr,    14);

    *ptr += 36;
    *ptr = *ptr + 25;
    *ptr = 25 + *ptr;
    BOOST_CHECK_EQUAL(*ptr,    100);
    BOOST_CHECK_EQUAL(*ptr, *ptr);

    img = ImageT::Pixel(111);
    BOOST_CHECK_EQUAL(img(0,0), 111);

    mask = MaskT::Pixel(0x666);
    BOOST_CHECK_EQUAL(mask(0,0), 0x666);
}

//
// Iterators
//
BOOST_AUTO_TEST_CASE(iterators) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    ImageT img = make_image();
    //
    // Count the pixels between begin() and end() (using a fast iterator)
    //
    {
        int i = 0;
        for (ImageT::fast_iterator ptr = img.begin(true), end = img.end(true); ptr != end; ++ptr, ++i) {
        }
        BOOST_CHECK_EQUAL(i, img.getWidth()*img.getHeight());
    }
    //
    // Count the pixels between begin() and end() (using a const_iterator)
    //
    {
        int i = 0;
        for (ImageT::const_iterator ptr = img.begin(), end = img.end(); ptr != end; ++ptr, ++i) {
        }
        BOOST_CHECK_EQUAL(i, img.getWidth()*img.getHeight());
    }
    //
    // Count the pixels between begin() and end() backwards (there is no const_reverse_iterator)
    //
    {
        int i = 0;
        for (ImageT::reverse_iterator ptr = img.rbegin(), end = img.rend(); ptr != end; ++ptr, ++i) {
        }
        BOOST_CHECK_EQUAL(i, img.getWidth()*img.getHeight());
    }
    //
    // Check begin() and our ability to increment it
    //
    {
        ImageT::iterator ptr = img.begin();
        ptr += img.getWidth() + 1;           // move to (1,1)

        BOOST_CHECK_EQUAL(*ptr, 101);
    }
    //
    // Check {col,row}_begin() and our ability to increment them
    //
    {
        ImageT::x_iterator rptr = img.row_begin(1);
        rptr += 1;                       // move to (1,1)

        BOOST_CHECK_EQUAL(*rptr, 101);

        BOOST_REQUIRE(img.getWidth() >= 4);
        ImageT::const_x_iterator at_ptr = img.x_at(3,2);
        BOOST_CHECK_EQUAL(*at_ptr, 302);
    }
    {
        ImageT::y_iterator cptr = img.col_begin(2);
        cptr += 1;                       // move to (2,1)

        BOOST_CHECK_EQUAL(*cptr, 201);

        BOOST_REQUIRE(img.getWidth() >= 4);
        ImageT::const_y_iterator at_ptr = img.y_at(3,2);
        BOOST_CHECK_EQUAL(*at_ptr, 302);
    }
    //
    // Test {col,row}_{begin,end} by finding the width and height
    //
    {
        int i = 0;
        for (ImageT::x_iterator ptr = img.row_begin(0), end = img.row_end(0); ptr != end; ++ptr) {
            ++i;
        }
        BOOST_CHECK_EQUAL(i, img.getWidth());

        i = 0;
        for (ImageT::y_iterator ptr = img.col_begin(0), end = img.col_end(0); ptr != end; ++ptr) {
            ++i;
        }
        BOOST_CHECK_EQUAL(i, img.getHeight());
    }
    //
    // Check that at() works too
    //
    {
        ImageT::const_iterator ptr = img.at(1,1);

        BOOST_CHECK_EQUAL(*ptr, 101);
    }
    //
    // Check that adding to the iterators works
    //
    {
        ImageT::x_iterator begin = img.row_begin(0), end = begin + img.getWidth();
        BOOST_CHECK(begin != end);
    }
    {
        ImageT::y_iterator end = img.col_end(0), begin = end - img.getWidth();
        BOOST_CHECK(!(begin == end));
    }
}

//
// Locators
//
BOOST_AUTO_TEST_CASE(locators) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    ImageT img = make_image();

    {
        ImageT::xy_locator loc = img.xy_at(1,1);

        BOOST_CHECK_EQUAL(*loc, 101);

        BOOST_CHECK_EQUAL(loc(1,1), 202);

        loc += image::pair2I(-1, 1);    // loc == img.xy_at(0, 2);
        BOOST_CHECK_EQUAL(*loc, 2);

        loc.x() += 2;
        ++loc.x();        // loc == img.xy_at(3, 2);
        loc.y() += 1;
        loc.y() += 1;     // loc == img.xy_at(3, 4);
        BOOST_REQUIRE(img.getWidth() >= 4);
        BOOST_REQUIRE(img.getHeight() >= 5);

        BOOST_CHECK_EQUAL(*loc, 304);

        BOOST_CHECK_EQUAL(*loc.x(), 304);

        BOOST_CHECK_EQUAL(*loc.y(), 304);

        ++loc.x();
        BOOST_CHECK_EQUAL(*loc.x(), 404);
        BOOST_CHECK_EQUAL(*loc, 404);
    }

    {
        ImageT::xy_locator loc = img.xy_at(1,1);
        ImageT::xy_locator::cached_location_t above = loc.cache_location(0,  1);
        ImageT::xy_locator::cached_location_t below = loc.cache_location(0, -1);

        BOOST_CHECK_EQUAL(loc[above], 102);
        BOOST_CHECK_EQUAL(loc[below], 100);
    }
}
