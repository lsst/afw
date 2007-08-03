// -*- lsst-c++ -*-
#include "lsst/fw/Mask.h"
#include "lsst/fw/LSSTFitsResource.h"
#include "lsst/mwi/data/DataProperty.h"

#include <stdexcept>

using namespace lsst::fw;
using boost::any_cast;

using lsst::mwi::data::DataProperty;

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
    DataProperty::PtrType metaDataPtr = testMask.getMetaData();
    std::cout << metaDataPtr->toString("",true) << std::endl;

    // try some pattern matching on metadata
    const std::string pattern("WAT.*");
    std::cout << "Searching metadata with pattern " + pattern << std::endl;
    DataProperty::iteratorRangeType matches = metaDataPtr->searchAll(pattern);
    DataProperty::ContainerType::const_iterator iter;
    for( iter = matches.first; iter!= matches.second; iter++) {
        std::cout << "    found " + (*iter)->toString() << std::endl;
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
        } catch (lsst::mwi::exceptions::Exception &e) {
            throw lsst::mwi::exceptions::Exception(std::string("In handler\n") + e.what());
        }
    } catch (lsst::mwi::exceptions::Exception &e) {
        std::clog << e.what() << endl;
    }
}
