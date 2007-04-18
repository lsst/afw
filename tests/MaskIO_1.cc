// -*- lsst-c++ -*-
#include "lsst/fw/Mask.h"
#include "lsst/fw/LSSTFitsResource.h"

#include <stdexcept>

using namespace lsst;
using boost::any_cast;


/*
 * Make this a subroutine so that locals go out of scope as part of test
 * of memory management
 */
void test() {

    // NOTE:  does not work with <uint16> until DiskResourceFITS bug is fixed
    typedef PixelGray<float> MaskPixelType;

    Mask<MaskPixelType> testMask;
    testMask.readFits("filename.fits");

    // check whether Mask planes got setup right from FITS header...
    std::cout << "MaskPlanes from FITS header:" << std::endl;
    testMask.printMaskPlanes();

    // check the full metadata from the FITS header
    DataProperty::DataPropertyPtrT metaDataPtr = testMask.getMetaData();
    metaDataPtr->print();
    std::cout << std::endl;

    // try some pattern matching on metadata
    DataProperty::DataPropertyPtrT matchPtr = metaDataPtr->find(boost::regex("WAT.*"));
    while (matchPtr) {
        matchPtr->print();
        matchPtr = metaDataPtr->find(boost::regex("WAT.*"), false);
    }
}

int main(int argc, char *argv[]) {
    try {
        try {
            test();
        } catch (lsst::Exception &e) {
            throw lsst::Exception(std::string("In handler\n") + e.what());
        }
    } catch (lsst::Exception &e) {
        std::clog << e.what() << endl;
    }
}
