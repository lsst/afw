// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <typeinfo>
#include <cstdio>

#include "lsst/daf/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image.h"

using namespace std;
using lsst::pex::logging::Trace;
namespace pexEx = lsst::pex::exceptions;
namespace image = lsst::afw::image;
namespace geom = lsst::afw::geom;

int test(int argc, char**argv) {
    if (argc < 5) {
       cerr << "Usage: inputBaseName1 inputBaseName2 outputBaseName1  outputBaseName2" << endl;
       return EXIT_FAILURE;
    }
    
    Trace::setDestination(cout);
    Trace::setVerbosity(".", 0);
    
    typedef image::MaskedImage<float> MaskedImage;

    //
    // We want to construct the MaskedImage within a try block, so declare a pointer outside
    //
    MaskedImage::Ptr testMaskedImage1;
    try {
        testMaskedImage1 = MaskedImage::Ptr(new MaskedImage(argv[1]));
    } catch (pexEx::Exception &e) {
        cerr << "Failed to open " << argv[1] << ": " << e.what() << endl;
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
        testFlat = MaskedImage::Ptr(new MaskedImage(argv[2]));
    } catch (pexEx::Exception &e) {
        cerr << "Failed to open " << argv[2] << ": " << e.what() << endl;
        return EXIT_FAILURE;
    }
    *testFlat->getVariance() = 20.0;

    *testFlat /= 20000.0;

    testMaskedImage2 *= *testFlat;

    // test of fits write

    testMaskedImage2.writeFits(argv[3]);

    // test of subImage

    geom::Box2I region(geom::Point2I(100, 600), geom::Extent2I(200, 300));
    MaskedImage subMaskedImage1 = MaskedImage(
        *testMaskedImage1, 
        region,
        image::LOCAL
    );
    subMaskedImage1 *= 0.5;
    subMaskedImage1.writeFits(argv[4]);

    // Check whether offsets have been correctly saved
    geom::Box2I region2(geom::Point2I(80, 110), geom::Extent2I(20, 30));
    MaskedImage subMaskedImage2 = MaskedImage(subMaskedImage1, region2, image::LOCAL);

    cout << "Offsets: " << subMaskedImage2.getX0() << " " << subMaskedImage2.getY0() << endl;

    testMaskedImage1->writeFits(argv[3]);
    
    std::remove(argv[3]);
    std::remove(argv[4]);
    
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
