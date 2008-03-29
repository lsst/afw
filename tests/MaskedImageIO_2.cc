// -*- lsst-c++ -*-
#include <lsst/daf/data.h>
#include <lsst/pex/logging/Trace.h>
#include <lsst/afw/image.h>
#include <lsst/afw/math.h>

#include <stdexcept>

using namespace std;
using boost::any_cast;

using lsst::pex::logging::Trace;
using lsst::daf::data::DataProperty;
using lsst::daf::data::FitsFormatter;

namespace pexEx = lsst::pex::exceptions;

/*
 * Make this a subroutine so that locals go out of scope as part of test
 * of memory management
 */
void test(char *name) {

    typedef uint16 MaskPixelType;
    typedef float ImagePixelType;

    lsst::afw::image::Mask<MaskPixelType>::MaskPlaneDict LSSTPlaneDefs;

    LSSTPlaneDefs["BAD"] = 0;
    LSSTPlaneDefs["SAT"] = 1;
    LSSTPlaneDefs["EDGE"] = 2;
    LSSTPlaneDefs["OBJ"] = 3;

    lsst::afw::image::MaskedImage<ImagePixelType, MaskPixelType> testMasked(LSSTPlaneDefs);
    testMasked.readFits(name, true);   // second arg says to conform mask planes

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
