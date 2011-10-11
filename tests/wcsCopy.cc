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
 * Test wcs copy constructor and assignment operator.
 */
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>

#include "lsst/daf/base.h"
#include "lsst/afw/image.h"

using lsst::daf::base::PropertySet;

void doTest(lsst::afw::image::Wcs::Ptr wcsPtr);


int main() {
    typedef float Pixel;

    { //memory (de)allocation block
        char *afwdataCStr = getenv("AFWDATA_DIR");
        if (afwdataCStr == 0) {
            std::cout << "afwdata must be set up" << std::endl;
            exit(1);
        }
        std::string afwdata(afwdataCStr);
        const std::string inFilename(afwdata + "/small_MI");

        // Create a wcs from a fits file (so the wcs has some memory to allocate)
        std::cout << "Opening file " << inFilename << std::endl;

        PropertySet::Ptr metadata(new PropertySet);
        int const hdu = 0;
        lsst::afw::image::MaskedImage<Pixel, lsst::afw::image::MaskPixel> mImage(inFilename, hdu, metadata);
        
        //Test the TanWcs class
        lsst::afw::image::Wcs::Ptr  wcsPtr = lsst::afw::image::makeWcs(metadata);
        doTest(wcsPtr);
        
        //Change the CTYPES so makeWcs creates an object of the base class
        metadata->set("CTYPE1", "RA---SIN");
        metadata->set("CTYPE2", "DEC--SIN");
        wcsPtr = lsst::afw::image::makeWcs(metadata);
        doTest(wcsPtr);
        
    } // close memory (de)allocation block

    // check for memory leaks
    if (lsst::daf::base::Citizen::census(0) != 0) {
        std::cerr << "Leaked memory blocks:" << std::endl;
        lsst::daf::base::Citizen::census(std::cerr);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}



void doTest(lsst::afw::image::Wcs::Ptr wcsPtr) {

    std::cout << "making a copy of a wcs" << std::endl;
    { // use copy constructor and deallocate the copy
        lsst::afw::image::Wcs::Ptr wcsCopy = wcsPtr->clone();
        lsst::afw::image::Wcs::Ptr wcsCopy2 = wcsCopy->clone();
    }
    std::cout << "deallocated the copy; assigning a wcs" << std::endl;
}
