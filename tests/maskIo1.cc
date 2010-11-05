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

#include "lsst/daf/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image.h"

using namespace std;
using lsst::daf::base::PropertySet;
namespace afwImage = lsst::afw::image;

/*
 * Make this a subroutine so that locals go out of scope as part of test
 * of memory management
 */
void test(char *name) {

    typedef afwImage::MaskPixel MaskPixel;

    PropertySet::Ptr metadata(new PropertySet());
    
    afwImage::Mask<MaskPixel> testMask(name, 0, metadata);

    // check whether Mask planes got setup right from FITS header...
    cout << "MaskPlanes from FITS header:" << endl;
    testMask.printMaskPlanes();

    // check the full metadata from the FITS header
    cout << metadata->toString() << endl;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: inputBaseName" << endl;
        return EXIT_FAILURE;
    }

    try {
        test(argv[1]);
    } catch (std::exception const &e) {
        clog << e.what() << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
