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

#include "lsst/afw/image/MaskedImage.h"

const std::string outFile("rwfitsOut");

int main(int argc, char **argv) {

    std::string file;
    if (argc == 2) {
        file = std::string(argv[1]);
    } else {
        std::string afwdata = getenv("AFWDATA_DIR");
        if (afwdata.empty()) {
            std::cerr << "Usage: maskedImageFitsIO fitsFile" << std::endl;
            std::cerr << "fitsFile excludes the \"_img.fits\" suffix" << std::endl;
            std::cerr << "AFWDATA_DIR not set.  Provide fits file as argument or setup afwdata.\n"
                      << std::endl;
            exit(EXIT_FAILURE);
        } else {
            file = afwdata + "/small_MI";
        }
    }
    std::cout << "Running with: " <<  file << std::endl;

    lsst::afw::image::MaskedImage<float> mImage(file);
    mImage.writeFits(outFile);
}
