// -*- lsst-c++ -*-
#include <stdexcept>

#include <lsst/pex/exceptions.h>
#include <lsst/daf/data.h>
#include <lsst/afw/image.h>

using namespace std;
using lsst::daf::base::DataProperty;
namespace afwImage = lsst::afw::image;

/*
 * Make this a subroutine so that locals go out of scope as part of test
 * of memory management
 */
void test(char *name) {

    typedef afwImage::maskPixelType maskPixelType;

    afwImage::Mask<maskPixelType> testMask;
    testMask.readFits(name);

    // check whether Mask planes got setup right from FITS header...
    cout << "MaskPlanes from FITS header:" << endl;
    testMask.printMaskPlanes();

    // check the full metadata from the FITS header
    DataProperty::PtrType metaDataPtr = testMask.getMetaData();
    cout << metaDataPtr->toString("",true) << endl;

    // try some pattern matching on metadata
    const string pattern("WAT.*");
    cout << "Searching metadata with pattern " + pattern << endl;
    DataProperty::iteratorRangeType matches = metaDataPtr->searchAll(pattern);
    DataProperty::ContainerType::const_iterator iter;
    for( iter = matches.first; iter!= matches.second; iter++) {
        cout << "    found " + (*iter)->toString() << endl;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: inputBaseName" << endl;
        return 1;
    }

    try {
        try {
            test(argv[1]);
        } catch (lsst::pex::exceptions::ExceptionStack &e) {
            throw lsst::pex::exceptions::Runtime(string("In handler\n") + e.what());
        }
    } catch (lsst::pex::exceptions::ExceptionStack &e) {
        clog << e.what() << endl;
    }
}
