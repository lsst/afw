// -*- lsst-c++ -*-
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
