// -*- lsst-c++ -*-
#include <stdexcept>

#include "lsst/daf/base.h"
#include "lsst/daf/data/FitsFormatter.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image.h"
#include "lsst/afw/math.h"

using namespace std;
using boost::any_cast;

using lsst::pex::logging::Trace;
using lsst::daf::base::DataProperty;
using lsst::daf::data::FitsFormatter;

namespace pexEx = lsst::pex::exceptions;
namespace image = lsst::afw::image;

/*
 * Make this a subroutine so that locals go out of scope as part of test
 * of memory management
 */
void test(char *name) {

    typedef image::MaskPixel MaskPixelType;
    typedef float ImagePixelType;

    const int hdu = 0;
    lsst::daf::base::DataProperty::PtrType metaData(static_cast<lsst::daf::base::DataProperty *>(NULL));
    bool const conformMask = true;      // use mask definitions from the file
    image::MaskedImage<ImagePixelType, MaskPixelType> testMasked(name, hdu, metaData, conformMask);

    testMasked.writeFits("testout");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: inputBaseName" << endl;
        return 1;
    }
    
    Trace::setDestination(cout);
    Trace::setVerbosity(".", 1);

    try {
        try {
            test(argv[1]);
        } catch (pexEx::ExceptionStack &e) {
            throw pexEx::Runtime(std::string("In handler\n") + e.what());
        }
    } catch (pexEx::ExceptionStack &e) {
        clog << e.what() << endl;
        return 1;
    }
}
