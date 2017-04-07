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

/*
 * MaskedImage iterator tutorial.
 */

// Include the necessary headers;
// if using many image modules then you may prefer to include "lsst/afw/image.h"
#include "lsst/afw/image/MaskedImage.h"

// Declare the desired MaskedImage type.
// Note: only specific types are supported; for the list of available types
// see the explicit instantiation code at the end of lsst/afw/image/src/MaskedImage.cc
namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;
typedef afwImage::MaskedImage<int> ImageT;

int main() {

    // Declare a MaskedImage; its pixels are not yet initialized.
    ImageT img(afwGeom::Extent2I(10, 6));

    // Initialize all pixels to a given value.
    img = ImageT::Pixel(100, 0x1, 10);

    // Here is a common and efficient way to set all pixels of the image.
    // Note that the end condition is only computed once, for efficiency.
    for (int y = 0; y != img.getHeight(); ++y) {
        for (ImageT::x_iterator ptr = img.row_begin(y), end = img.row_end(y); ptr != end; ++ptr) {
            *ptr = ImageT::Pixel(100, 0x1, 10);

            // Or, if you prefer, you may set image, mask and variance separately with no loss of speed
            ptr.image() = 100;
            ptr.mask() = 0x1;
            ptr.variance() = 10;
        }
    }

    // It is probably slower to compute the end condition each time, as is done here.
    for (int y = 0; y != img.getHeight(); ++y) {
        for (ImageT::x_iterator ptr = img.row_begin(y); ptr != img.row_end(y); ++ptr) {
            *ptr = ImageT::Pixel(100, 0x1, 10);
        }
    }

    // STL-compliant iterators are available.
    // However, they are not very efficient because the image data may not be contiguous
    // so these iterators must test for end-of-row on every increment.
    // (By the way, we do guarantee that an image's row data is contiguous).
    // iterator
    for (ImageT::iterator ptr = img.begin(), end = img.end(); ptr != end; ++ptr) {
        *ptr = ImageT::Pixel(100, 0x1, 10);
    }
    // reverse_iterator
    for (ImageT::reverse_iterator ptr = img.rbegin(), end = img.rend(); ptr != end; ++ptr) {
        *ptr = ImageT::Pixel(100, 0x1, 10);
    }
    // A different way of choosing begin() for use with (inefficient) iterator
    for (ImageT::iterator ptr = img.at(0, 0), end = img.end(); ptr != end; ++ptr) {
        *ptr = ImageT::Pixel(100, 0x1, 10);
    }

    // There is one efficient STL-compliant iterator: "fast_iterator", but it only works for contiguous images
    // (such as newly allocated images). If you attempt to use this on a subimage you will get an exception.
    for (ImageT::fast_iterator ptr = img.begin(true), end = img.end(true); ptr != end; ++ptr) {
        *ptr = ImageT::Pixel(100, 0x1, 10);
    }

    // It is possible to traverse the image by columns instead of by rows,
    // but because the data is row-contiguous, this has awful consequences upon cache performance.
    for (int x = 0; x != img.getWidth(); ++x) {
        for (ImageT::y_iterator ptr = img.col_begin(x), end = img.col_end(x); ptr != end; ++ptr) {
            *ptr = ImageT::Pixel(100, 0x1, 10);
        }
    }

    // If you must traverse the image by columns then consider doing it in batches to improve
    // cache performance, as shown here:
    int x = 0;
    for (; x != img.getWidth()%4; ++x) {
        for (ImageT::y_iterator ptr = img.col_begin(x), end = img.col_end(x); ptr != end; ++ptr) {
            *ptr = ImageT::Pixel(100, 0x1, 10);
        }
    }
    for (; x != img.getWidth(); x += 4) {
        for (ImageT::y_iterator ptr0 = img.col_begin(x+0), end0 = img.col_end(x+0),
                                ptr1 = img.col_begin(x+1),
                                ptr2 = img.col_begin(x+2),
                                ptr3 = img.col_begin(x+3);
             ptr0 != end0; ++ptr0, ++ptr1, ++ptr2, ++ptr3) {
            *ptr0 = *ptr1 = *ptr2 = *ptr3 = ImageT::Pixel(100, 0x1, 10);
        }
    }

    // Save the image to disk
    img.writeFits("foo");

    return 0;
}
