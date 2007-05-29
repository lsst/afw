// -*- lsst-c++ -*-
#include "lsst/fw/MaskedImage.h"
#include "lsst/fw/Trace.h"

#include <stdexcept>

using namespace lsst::fw;
using boost::any_cast;


/*
 * Make this a subroutine so that locals go out of scope as part of test
 * of memory management
 */
void test(char *name) {

    typedef uint16 MaskPixelType;
    typedef float ImagePixelType;

    MaskedImage<ImagePixelType, MaskPixelType> testMasked;
    testMasked.readFits(name);

}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: inputBaseName" << std::endl;
        return 1;
    }
    
    Trace::setDestination(std::cout);
    Trace::setVerbosity(".", 1);

    try {
        try {
            test(argv[1]);
        } catch (lsst::fw::Exception &e) {
            throw lsst::fw::Exception(std::string("In handler\n") + e.what());
        }
    } catch (lsst::fw::Exception &e) {
        std::clog << e.what() << endl;
        return 1;
    }
}
