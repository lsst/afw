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

#include "lsst/utils/Utils.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/MaskedImage.h"

const std::string outImagePath("rwfitsOut.fits");

int main(int argc, char **argv) {

    std::string inImagePath;
    if (argc == 2) {
        inImagePath = std::string(argv[1]);
    } else {
        try {
            std::string dataDir = lsst::utils::getPackageDir("afwdata");
            inImagePath = dataDir + "/data/small.fits";
        } catch (lsst::pex::exceptions::NotFoundError) {
            std::cerr << "Usage: maskedImageFitsIO [fitsFile]" << std::endl;
            std::cerr << "fitsFile is the path to a masked image" << std::endl;
            std::cerr << "\nError: setup afwdata or specify fitsFile.\n" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Running with: " <<  inImagePath << std::endl;

    lsst::afw::image::MaskedImage<float> mImage(inImagePath);
    mImage.writeFits(outImagePath);
    std::cout << "Wrote masked image: " << outImagePath << std::endl;
}
