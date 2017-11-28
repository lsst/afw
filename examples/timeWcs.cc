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
#include <cmath>

#include "lsst/utils/Utils.h"
#include "lsst/afw/image.h"
#include "lsst/afw/geom.h"

const unsigned DefNIter = 100000;

/**
@internal Transform pix to sky and back again nIter times using points distributed evenly
from bbox.getMin() to bbox.getMax() and return the max round trip pixel error
*/
void timeWcs(lsst::afw::geom::SkyWcs const &wcs, lsst::afw::geom::Box2D &bbox, unsigned int nIter) {
    lsst::afw::geom::Extent2D maxErr;
    auto const dxy = bbox.getDimensions() / static_cast<float>(nIter);
    auto const xy0 = bbox.getMin();
    auto startClock = std::clock();
    for (unsigned int iter = 0; iter < nIter; ++iter) {
        auto pixPos = xy0 + dxy * static_cast<double>(iter);
        auto skyPos = wcs.pixelToSky(pixPos);
        auto retPixPos = wcs.skyToPixel(skyPos);
        for (int i = 0; i < 2; ++i) {
            maxErr[i] = std::fmax(maxErr[i], std::fabs(retPixPos[i] - pixPos[i]));
        }
    }
    // separate casts for CLOCKS_PER_SEC and nIter avoids incorrect results, perhaps due to overflow
    double usecPerIter = 1.0e6 * (clock() - startClock) /
                         (static_cast<double>(CLOCKS_PER_SEC) * static_cast<double>(nIter));
    std::cout << usecPerIter << " usec per iteration; max round trip error = " << maxErr << " pixels\n";
}

int main(int argc, char **argv) {
    std::string inImagePath;
    if (argc < 2) {
        try {
            std::string dataDir = lsst::utils::getPackageDir("afwdata");
            inImagePath = dataDir + "/ImSim/calexp/v85408556-fr/R23/S11.fits";
        } catch (lsst::pex::exceptions::NotFoundError) {
            std::cerr << "Usage: timeWcs [fitsFile [nIter]]" << std::endl;
            std::cerr << "fitsFile is the path to an exposure" << std::endl;
            std::cerr << "nIter (default " << DefNIter << ") is the number of iterations" << std::endl;
            std::cerr << "\nError: setup afwdata or specify fitsFile.\n" << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        inImagePath = std::string(argv[1]);
    }

    unsigned int nIter = DefNIter;
    if (argc > 2) {
        std::istringstream(argv[2]) >> nIter;
    }

    auto exposure = lsst::afw::image::Exposure<float>(inImagePath);
    auto bbox = lsst::afw::geom::Box2D(exposure.getBBox());
    if (!exposure.hasWcs()) {
        std::cerr << "Exposure " << inImagePath << " has no WCS\n" << std::endl;
        exit(EXIT_FAILURE);
    }
    auto wcsPtr = exposure.getWcs();

    std::cout << "Timing " << nIter << " iterations of pixel->sky->pixel of the WCS found in " << inImagePath
              << std::endl;

    timeWcs(*wcsPtr, bbox, nIter);
}
