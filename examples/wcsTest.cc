// -*- LSST-C++ -*- // fixed format comment for emacs

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
 * Simple test code for the Wcs Class
 */

#include <iostream>
#include <sstream>
#include <string>

#include "boost/format.hpp"
#include <memory>

#include "lsst/geom.h"
#include "lsst/utils/Utils.h"
#include "lsst/log/Log.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/afw/image.h"

/*
 * This test code incorporates some very simple tests of the Wcs class and related classes.
 */

namespace afwImage = lsst::afw::image;

using lsst::daf::base::PropertySet;

int main(int argc, char **argv) {
    typedef double Pixel;
    LOG_CONFIG();

    std::string inImagePath;
    if (argc < 2) {
        try {
            std::string dataDir = lsst::utils::getPackageDir("afwdata");
            inImagePath = dataDir + "/data/medexp.fits";
        } catch (lsst::pex::exceptions::NotFoundError) {
            std::cerr << "Usage: wcsTest [fitsFile]" << std::endl;
            std::cerr << "fitsFile is the path to an exposure" << std::endl;
            std::cerr << "\nError: setup afwdata or specify fitsFile.\n" << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        inImagePath = std::string(argv[1]);
    }
    std::cout << "Opening exposure " << inImagePath << std::endl;

    afwImage::Exposure<Pixel> exposure(inImagePath);
    if (!exposure.hasWcs()) {
        std::cerr << "Exposure does not have a WCS." << std::endl;
        exit(EXIT_FAILURE);
    }
    auto wcs = exposure.getWcs();

    // Testing input col, row values

    auto minCoord = lsst::geom::Point2D(1.0, 1.0);
    auto xy = lsst::geom::Point2D(exposure.getWidth(), exposure.getHeight());

    auto sky1 = wcs->pixelToSky(minCoord);
    auto sky2 = wcs->pixelToSky(xy);

    auto miRa1 = sky1.getLongitude();
    auto miDecl1 = sky1.getLatitude();
    auto miRa2 = sky2.getLongitude();
    auto miDecl2 = sky2.getLatitude();

    std::cout << "ra, decl of " << inImagePath << " at (" << minCoord[0] << " " << minCoord[1] << ") = "
              << "ra: " << miRa1.asDegrees() << " decl: " << miDecl1.asDegrees() << std::endl
              << std::endl;

    std::cout << "ra, decl of " << inImagePath << " at (" << xy[0] << " " << xy[1] << ") = "
              << "ra: " << miRa2.asDegrees() << " decl: " << miDecl2.asDegrees() << std::endl
              << std::endl;

    auto pixelScale0 = wcs->getPixelScale(minCoord);
    auto pixelScale1 = wcs->getPixelScale(xy);

    std::cout << "pixel scales: " << pixelScale0.asArcseconds() << ", " << pixelScale1.asArcseconds()
              << " arcsec/pixel\n";

    // Testing input ra, dec values using output from above for now

    auto pix1 = wcs->skyToPixel(sky1);
    auto pix2 = wcs->skyToPixel(sky2);

    std::cout << "col, row of " << inImagePath << " at (" << miRa1.asDegrees() << " " << miDecl1.asDegrees()
              << ") = "
              << "col: " << pix1[0] << " row: " << pix1[1] << std::endl
              << std::endl;

    std::cout << "col, row of " << inImagePath << " at (" << miRa2.asDegrees() << " " << miDecl2.asDegrees()
              << ") = "
              << "col: " << pix2[0] << " row: " << pix2[1] << std::endl
              << std::endl;

    auto raDecl1 = lsst::geom::SpherePoint(miRa1, miDecl1);
    auto raDecl2 = lsst::geom::SpherePoint(miRa2, miDecl2);

    auto pix3 = wcs->skyToPixel(raDecl1);
    auto pix4 = wcs->skyToPixel(raDecl2);

    std::cout << "col, row of " << inImagePath << " at " << raDecl1 << " = "
              << "col: " << pix3[0] << " row: " << pix3[1] << std::endl
              << std::endl;

    std::cout << "col, row of " << inImagePath << " at " << raDecl2 << " = "
              << "col: " << pix4[0] << " row: " << pix4[1] << std::endl
              << std::endl;
}
