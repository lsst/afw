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

#include "boost/format.hpp"

#include "boost/version.hpp"
#if BOOST_VERSION < 106900
#include "boost/gil/gil_all.hpp"
#else
#include "boost/gil.hpp"
#endif

template <class ImageT>
void timePixelAccess(ImageT const &image, float const pix, int nIter) {
    const int nCols = image.width();
    const int nRows = image.height();

    clock_t startTime = clock();
    for (int iter = 0; iter < nIter; ++iter) {
        for (int y = 0; y < image.height(); ++y) {
            for (typename ImageT::x_iterator ptr = image.row_begin(y), end = image.row_end(y); ptr != end;
                 ++ptr) {
                (*ptr)[0] += pix;
            }
        }
    }
    // separate casts for CLOCKS_PER_SEC and nIter avoids incorrect results, perhaps due to overflow
    double secPerIter =
            (clock() - startTime) / (static_cast<double>(CLOCKS_PER_SEC) * static_cast<double>(nIter));
    double const megaPix = static_cast<double>(nCols * nRows) / 1.0e6;
    std::cout << boost::format("Pixel Iterator\t%d\t%d\t%g\t%-8g\t%-8.1f") % nCols % nRows % megaPix %
                         secPerIter % (megaPix / secPerIter)
              << std::endl;

    startTime = clock();
    for (int iter = 0; iter < nIter; ++iter) {
        for (int y = 0; y < image.height(); ++y) {
            for (typename ImageT::xy_locator ptr = image.xy_at(0, y), end = image.xy_at(nCols, y); ptr != end;
                 ++ptr.x()) {
                (*ptr)[0] += pix;
            }
        }
    }
    secPerIter = (clock() - startTime) / (static_cast<double>(CLOCKS_PER_SEC) * static_cast<double>(nIter));
    std::cout << boost::format("Pixel Locator\t%d\t%d\t%g\t%-8g\t%-8.1f") % nCols % nRows % megaPix %
                         secPerIter % (megaPix / secPerIter)
              << std::endl;
}

int main(int argc, char **argv) {
    int const DefNIter = 100;
    int const DefNCols = 1024;

    if ((argc == 2) && (argv[1][0] == '-')) {
        std::cout << "Usage: timePixelAccess [nIter [nCols [nRows]]]" << std::endl;
        std::cout << "nIter (default " << DefNIter << ") is the number of iterations" << std::endl;
        std::cout << "nCols (default " << DefNCols << ") is the number of columns" << std::endl;
        std::cout << "nRows (default = nCols) is the number of rows" << std::endl;
        return 1;
    }

    int nIter = DefNIter;
    if (argc > 1) {
        std::istringstream(argv[1]) >> nIter;
    }
    int nCols = DefNCols;
    if (argc > 2) {
        std::istringstream(argv[2]) >> nCols;
    }
    int nRows = nCols;
    if (argc > 3) {
        std::istringstream(argv[3]) >> nRows;
    }

    std::cout << "Accessor Type\tCols\tRows\tMPix\tSecPerIter\tSecPerIterPerMPix" << std::endl;
    std::cout << "Image(" << nCols << ", " << nRows << ")" << std::endl;

    boost::gil::gray32f_image_t image(nCols, nRows);

    float pix = 1.0;
    timePixelAccess(boost::gil::view(image), pix, nIter);
}
