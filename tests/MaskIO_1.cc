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
    LSSTFitsResource inFits("filename.fits");
    DataProperty::DataPropertyPtrT metaDataPtr = inFits.getMetaData();
    metaDataPtr->print();
    std::cout << std::endl;
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
