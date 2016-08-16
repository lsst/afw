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
#include <sstream>
#include <ctime>
#include <valarray>

#include "lsst/afw/image.h"

namespace image = lsst::afw::image;
namespace geom = lsst::afw::geom;

int main(int argc, char **argv) {
    typedef image::Image<float> ImageT;

    int const DefNIter = 100;
    int const DefNCols = 1024;

    if ((argc == 2) && (argv[1][0] == '-')) {
        std::cout << "Usage: timeImageAddition [nIter [nCols [nRows]]]" << std::endl;
        std::cout << "nIter (default " << DefNIter << ") is the number of iterations" << std::endl;
        std::cout << "nCols (default " << DefNCols << ") is the number of columns" << std::endl;
        std::cout << "nRows (default = nCols) is the number of rows" << std::endl;
        return 1;
    }

    unsigned nIter = DefNIter;
    if (argc > 1) {
        std::istringstream(argv[1]) >> nIter;
    }
    unsigned nCols = DefNCols;
    if (argc > 2) {
        std::istringstream(argv[2]) >> nCols;
    }
    unsigned nRows = nCols;
    if (argc > 3) {
        std::istringstream(argv[3]) >> nRows;
    }

    ImageT image(geom::Extent2I(nCols, nRows));

    std::cout << "\tCols\tRows\tMPix\tSecPerIter\tMPix/sec" << std::endl;
    //
    // Use the STL iterators
    //
    clock_t startTime = clock();
    for (unsigned iter = 0; iter < nIter; ++iter) {
        for (ImageT::iterator ptr = image.begin(), end = image.end(); ptr != end; ++ptr) {
            *ptr += 1;
        }
    }

    double const megaPix = static_cast<double>(nCols * nRows) / 1.0e6;

    // separate casts for CLOCKS_PER_SEC and nIter avoids incorrect results, perhaps due to overflow
    double secPerIter = (clock() - startTime)/
        (static_cast<double>(CLOCKS_PER_SEC)*static_cast<double>(nIter));
    std::cout << "STL\t" << nCols << "\t" << nRows << "\t" << megaPix << "\t" << secPerIter << "\t\t" <<
        static_cast<double>(megaPix)/secPerIter << std::endl;
    //
    // Now per-row iterators
    //
    startTime = clock();
    for (unsigned iter = 0; iter < nIter; ++iter) {
        for (int y = 0; y != image.getHeight(); ++y) {
            for (ImageT::x_iterator ptr = image.row_begin(y), end = image.row_end(y); ptr != end; ++ptr) {
                *ptr += 1;
            }
        }
    }

    secPerIter = (clock() - startTime)/
        (static_cast<double>(CLOCKS_PER_SEC)*static_cast<double>(nIter));
    std::cout << "Per row\t" << nCols << "\t" << nRows << "\t" << megaPix << "\t" << secPerIter << "\t\t" <<
        static_cast<double>(megaPix)/secPerIter << std::endl;
    //
    // Use a fast STL compiliant iterator, but the pixel order's undefined
    //
    startTime = clock();
    for (unsigned iter = 0; iter < nIter; ++iter) {
        for (ImageT::fast_iterator ptr = image.begin(true), end = image.end(true); ptr != end; ++ptr){
            *ptr += 1;
        }
    }

    secPerIter = (clock() - startTime)/
        (static_cast<double>(CLOCKS_PER_SEC)*static_cast<double>(nIter));
    std::cout << "STL 2\t" << nCols << "\t" << nRows << "\t" << megaPix << "\t" << secPerIter << "\t\t" <<
        static_cast<double>(megaPix)/secPerIter << std::endl;
    //
    // Now copy and iterate over whole image
    //
    std::vector<ImageT::Pixel> vec(image.getWidth()*image.getHeight());
    std::vector<ImageT::Pixel>::iterator vptr = vec.begin();
    for (int y = 0; y != image.getHeight(); ++y) {
        std::copy(image.row_begin(y), image.row_end(y), vptr);
        vptr += image.getWidth();
    }
    startTime = clock();
    for (unsigned iter = 0; iter < nIter; ++iter) {
        for (std::vector<ImageT::Pixel>::iterator ptr = vec.begin(), end = vec.end(); ptr != end; ++ptr) {
            *ptr += 1;
        }
    }

    secPerIter = (clock() - startTime)/
        (static_cast<double>(CLOCKS_PER_SEC)*static_cast<double>(nIter));
    std::cout << "Vector\t" << nCols << "\t" << nRows << "\t" << megaPix << "\t" << secPerIter << "\t\t" <<
        static_cast<double>(megaPix)/secPerIter << std::endl;
#if 0
    std::cout << image(0,0) << " " << image(image.getWidth() - 1, image.getHeight() - 1) << std::endl;
    std::cout << vec[0] << " " << vec[image.getWidth()*image.getHeight() - 1] << std::endl;
#endif
    //
    // Try a valarray
    //
    std::valarray<ImageT::Pixel> varray(*image.begin(), image.getWidth()*image.getHeight());

    startTime = clock();
    for (unsigned iter = 0; iter < nIter; ++iter) {
        for (unsigned i = 0; i != varray.size(); ++i) {
            varray[i] += 1;
        }
    }

    secPerIter = (clock() - startTime)/
        (static_cast<double>(CLOCKS_PER_SEC)*static_cast<double>(nIter));
    std::cout << "Varray\t" << nCols << "\t" << nRows << "\t" << megaPix << "\t" << secPerIter << "\t\t" <<
        static_cast<double>(megaPix)/secPerIter << std::endl;
#if 0
    std::cout << image(0,0) << " " << image(image.getWidth() - 1, image.getHeight() - 1) << std::endl;
    std::cout << varray[0] << " " << varray[image.getWidth()*image.getHeight() - 1] << std::endl;
#endif
}
