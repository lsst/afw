// -*- lsst-c++ -*-
#include "lsst/fw/Mask.h"
#include "lsst/fw/LSSTFitsResource.h"

#include <stdexcept>

using namespace lsst::fw;
using boost::any_cast;


/*
 * Make this a subroutine so that locals go out of scope as part of test
 * of memory management
 */
void test(char *name) {

    typedef uint16 MaskPixelType;

    Mask<MaskPixelType> testMask;
    testMask.readFits(name);

    // check whether Mask planes got setup right from FITS header...
    std::cout << "MaskPlanes from FITS header:" << std::endl;
    testMask.printMaskPlanes();

    // check the full metadata from the FITS header
    DataPropertyPtrT metaDataPtr = testMask.getMetaData();
    metaDataPtr->print();
    std::cout << std::endl;

    // try some pattern matching on metadata
    DataPropertyPtrT matchPtr = metaDataPtr->find(boost::regex("WAT.*"));
    while (matchPtr) {
        matchPtr->print();
        matchPtr = metaDataPtr->find(boost::regex("WAT.*"), false);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: inputBaseName" << std::endl;
        return 1;
    }

    try {
        try {
            test(argv[1]);
        } catch (lsst::fw::Exception &e) {
            throw lsst::fw::Exception(std::string("In handler\n") + e.what());
        }
    } catch (lsst::fw::Exception &e) {
        std::clog << e.what() << endl;
    }
}
