// -*- lsst-c++ -*-
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

    afwImage::Mask<afwImage::MaskPixel> testMask;
    testMask.readFits(name);

    afwImage::Mask<MaskPixel> masterMask;
    masterMask.readFits(masterName);

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
