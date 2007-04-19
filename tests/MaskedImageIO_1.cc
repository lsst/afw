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
void test() {

    // NOTE:  does not work with <uint16> until DiskResourceFITS bug is fixed
    typedef PixelGray<float> MaskPixelType;
    typedef PixelGray<float> ImagePixelType;

    MaskedImage<ImagePixelType, MaskPixelType> testMasked;
    testMasked.readFits("imageBase");

}

int main(int argc, char *argv[]) {

    using namespace lsst::fw::Trace;

    setDestination(std::cout);

    setVerbosity(".", 1);

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
