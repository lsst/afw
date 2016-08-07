// -*- lsst-c++ -*-

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

#include <stdexcept>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image.h"

using namespace std;
namespace afwImage = lsst::afw::image;

/*
 * Make this a subroutine so that locals go out of scope as part of test
 * of memory management
 */
void test(char *name, char *masterName) {

    typedef afwImage::MaskPixel MaskPixel;

    afwImage::Mask<afwImage::MaskPixel> testMask(name);

    afwImage::Mask<MaskPixel> masterMask(masterName);

    // check whether Mask planes got setup right from FITS header...
    cout << "test MaskPlanes from FITS header:" << endl;
    testMask.printMaskPlanes();

    // check whether Mask planes got setup right from FITS header...
    cout << "master MaskPlanes from FITS header:" << endl;
    masterMask.printMaskPlanes();

    testMask.conformMaskPlanes(masterMask.getMaskPlaneDict());

    // check whether Mask planes got setup right from FITS header...
    cout << "test MaskPlanes from FITS header:" << endl;
    testMask.printMaskPlanes();

    // check whether Mask planes got mapped right to conform with master
    cout << "test MaskPlanes after conformMaskPlanes:" << endl;
    testMask.printMaskPlanes();

    testMask.writeFits("test_msk.fits");



}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: inputBaseName masterBaseName" << endl;
        return EXIT_FAILURE;
    }

    try {
        test(argv[1], argv[2]);
    } catch (std::exception const &e) {
        clog << e.what() << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
