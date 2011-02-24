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
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image.h"

using namespace std;

using lsst::pex::logging::Trace;
using lsst::daf::base::PropertySet;

namespace pexEx = lsst::pex::exceptions;
namespace image = lsst::afw::image;

/*
 * Make this a subroutine so that locals go out of scope as part of test
 * of memory management
 */
void test(char *name) {

    typedef image::MaskPixel MaskPixelType;
    typedef float ImagePixelType;

    int const hdu = 0;
    geom::BoxI bbox;
    PropertySet::Ptr metadata;
    bool const conformMask = true;      // use mask definitions from the file
    image::MaskedImage<ImagePixelType, MaskPixelType> testMasked(string(name), hdu,
                                                                 metadata, bbox, conformMask);

    testMasked.writeFits("testout");
    ::unlink("testout_img.fits");
    ::unlink("testout_msk.fits");
    ::unlink("testout_var.fits");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: inputBaseName" << endl;
        return EXIT_FAILURE;
    }

    int status = EXIT_SUCCESS;
    Trace::setDestination(cout);
    Trace::setVerbosity(".", 1);
    try {
        test(argv[1]);
    } catch (pexEx::Exception &e) {
        clog << e.what() << endl;
        status = EXIT_FAILURE;
    }

    // Check for memory leaks
    if (lsst::daf::base::Citizen::census(0) == 0) {
        cerr << "No leaks detected" << endl;
    } else {
        cerr << "Leaked memory blocks:" << endl;
        lsst::daf::base::Citizen::census(cerr);
        status = EXIT_FAILURE;
    }
    return status;
}
