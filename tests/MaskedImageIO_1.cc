// -*- lsst-c++ -*-
#include "lsst/fw/MaskedImage.h"
#include "lsst/fw/Trace.h"

#include <stdexcept>

using namespace lsst;
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
    fw::Trace::setDestination(std::cout);
    fw::Trace::setVerbosity(".", 1);

    try {
        try {
            test(argv[1]);
        } catch (lsst::Exception &e) {
            throw lsst::Exception(std::string("In handler\n") + e.what());
        }
    } catch (lsst::Exception &e) {
        std::clog << e.what() << endl;
    }
}
