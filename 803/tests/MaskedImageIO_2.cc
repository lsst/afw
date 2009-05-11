// -*- lsst-c++ -*-
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
    image::BBox bbox;
    PropertySet::Ptr metadata;
    bool const conformMask = true;      // use mask definitions from the file
    image::MaskedImage<ImagePixelType, MaskPixelType> testMasked(string(name), hdu, metadata, bbox, conformMask);

    testMasked.writeFits("testout");
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
