// -*- lsst-c++ -*-
#include "lsst/fw/MaskedImage.h"
#include "lsst/fw/WCS.h"
#include "lsst/mwi/utils/Trace.h"
#include "lsst/mwi/data/FitsFormatter.h"

#include <stdexcept>

using namespace std;
using namespace lsst::fw;
using boost::any_cast;

using lsst::mwi::utils::Trace;
using lsst::mwi::data::DataProperty;
using lsst::mwi::data::FitsFormatter;

namespace mwie = lsst::mwi::exceptions;

/*
 * Make this a subroutine so that locals go out of scope as part of test
 * of memory management
 */
void test(char *name) {

    typedef uint16 MaskPixelType;
    typedef float ImagePixelType;

    Mask<MaskPixelType>::MaskPlaneDict LSSTPlaneDefs;

    LSSTPlaneDefs[0] = "BAD";
    LSSTPlaneDefs[1] = "SAT";
    LSSTPlaneDefs[2] = "EDGE";
    LSSTPlaneDefs[3] = "OBJ";


    MaskedImage<ImagePixelType, MaskPixelType> testMasked(LSSTPlaneDefs);
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
        } catch (mwie::ExceptionStack &e) {
            throw mwie::Runtime(std::string("In handler\n") + e.what());
        }
    } catch (mwie::ExceptionStack &e) {
        clog << e.what() << endl;
        return 1;
    }
}
