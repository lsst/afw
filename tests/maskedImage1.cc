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
 
#include <typeinfo>
#include <cstdio>

#include "lsst/daf/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image.h"
#include "lsst/utils/Utils.h"

using namespace std;
using lsst::pex::logging::Trace;
namespace pexEx = lsst::pex::exceptions;
namespace image = lsst::afw::image;
namespace geom = lsst::afw::geom;

int test(int argc, char**argv) {
    
    string dataDir, inImagePath1, inImagePath2, outImagePath1, outImagePath2;
    if (argc < 2) {
        try {
            dataDir = lsst::utils::getPackageDir("afwdata");
            inImagePath1 = dataDir + "/data/871034p_1_MI.fits";
            inImagePath2 = dataDir + "/data/871034p_1_MI.fits"; // afw/tests/SConscript passes the same file twice in the previous avatar.
            outImagePath1 = "tests/file:maskedImage1_output_1.fits";
            outImagePath2 = "tests/file:maskedImage1_output_2.fits";
        } catch (lsst::pex::exceptions::NotFoundError) {
            cerr << "Usage: convolveGPU [fitsFile]" << endl;
            cerr << "Warning: tests not run! Setup afwdata if you wish to use the default fitsFile." << endl;
            return EXIT_SUCCESS;
        }
    }
    else {
        inImagePath1 = string(argv[1]);
        inImagePath2 = string(argv[2]);
        outImagePath1 = string(argv[3]);
        outImagePath2 = string(argv[4]);
    }
    
    Trace::setDestination(cout);
    Trace::setVerbosity(".", 0);
    
    typedef image::MaskedImage<float> MaskedImage;

    //
    // We want to construct the MaskedImage within a try block, so declare a pointer outside
    //
    MaskedImage::Ptr testMaskedImage1;
    try {
        testMaskedImage1 = MaskedImage::Ptr(new MaskedImage(inImagePath1));
    } catch (pexEx::Exception &e) {
        cerr << "Failed to open " << inImagePath1 << ": " << e.what() << endl;
        return EXIT_FAILURE;
    }

    *testMaskedImage1->getVariance() = 10.0;
    testMaskedImage1->getMask()->addMaskPlane("CR");
    
    // verify that copy constructor and operator= build and do not leak
    MaskedImage::Image testImage(geom::Extent2I(100, 100));
    MaskedImage::Image imageCopy(testImage);
    imageCopy = testImage;

    MaskedImage testMaskedImage2(testMaskedImage1->getDimensions()); // n.b. could just do a deep copy
    testMaskedImage2 = *testMaskedImage1;

    MaskedImage::Ptr testFlat;
    try {
        testFlat = MaskedImage::Ptr(new MaskedImage(inImagePath2));
    } catch (pexEx::Exception &e) {
        cerr << "Failed to open " << inImagePath2 << ": " << e.what() << endl;
        return EXIT_FAILURE;
    }
    *testFlat->getVariance() = 20.0;

    *testFlat /= 20000.0;

    testMaskedImage2 *= *testFlat;

    // test of fits write

    testMaskedImage2.writeFits(outImagePath1);

    // test of subImage

    geom::Box2I region(geom::Point2I(100, 600), geom::Extent2I(200, 300));
    MaskedImage subMaskedImage1 = MaskedImage(
        *testMaskedImage1, 
        region,
        image::LOCAL
    );
    subMaskedImage1 *= 0.5;
    subMaskedImage1.writeFits(outImagePath2);

    // Check whether offsets have been correctly saved
    geom::Box2I region2(geom::Point2I(80, 110), geom::Extent2I(20, 30));
    MaskedImage subMaskedImage2 = MaskedImage(subMaskedImage1, region2, image::LOCAL);

    cout << "Offsets: " << subMaskedImage2.getX0() << " " << subMaskedImage2.getY0() << endl;

    testMaskedImage1->writeFits(outImagePath1);
    
    std::remove(outImagePath1.c_str());
    std::remove(outImagePath2.c_str());
    
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    int status = EXIT_SUCCESS;
    try {
       status = test(argc, argv);
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
