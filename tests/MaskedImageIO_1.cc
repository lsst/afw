// -*- lsst-c++ -*-
#include "lsst/fw/MaskedImage.h"
#include "lsst/fw/WCS.h"
#include "lsst/mwi/utils/Trace.h"

#include <stdexcept>

using namespace lsst::fw;
using boost::any_cast;

using lsst::mwi::utils::Trace;
using lsst::mwi::data::DataPropertyPtrT;

namespace mwie = lsst::mwi::exceptions;

/*
 * Make this a subroutine so that locals go out of scope as part of test
 * of memory management
 */
void test(char *name) {

    typedef uint16 MaskPixelType;
    typedef float ImagePixelType;

    MaskedImage<ImagePixelType, MaskPixelType> testMasked;
    testMasked.readFits(name);

    DataPropertyPtrT metaDataPtr = testMasked.getImage()->getMetaData();

    std::ostringstream metaDataRepr;
    int nItems = 0;

    metaDataPtr->reprCfitsio(metaDataRepr, &nItems, false);
    
    Trace("MaskedImageIO_1", 1,
          boost::format("Number of FITS header cards: %d") % nItems);

    Trace("MaskedImageIO_1", 3,
          boost::format("FITS metadata string: %s") % metaDataRepr.str());

    WCS testWCS(metaDataPtr);

    Coord2D pix, sky;

//     pix[0] = testMasked.getCols() / 2.0;
//     pix[1] = testMasked.getRows() / 2.0;
    pix[0] = 500.0;
    pix[1] = 1000.0;

    testWCS.colRowToRaDec(pix, sky);

    Trace("MaskedImageIO_1", 1,
          boost::format("pix: %lf %lf") % pix[0] % pix[1]);

    Trace("MaskedImageIO_1", 1,
          boost::format("sky: %lf %lf") % sky[0] % sky[1]);

    testWCS.raDecToColRow(sky, pix);

    Trace("MaskedImageIO_1", 1,
          boost::format("pix: %lf %lf") % pix[0] % pix[1]);


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
        } catch (mwie::Exception &e) {
            throw mwie::Exception(std::string("In handler\n") + e.what());
        }
    } catch (mwie::Exception &e) {
        std::clog << e.what() << endl;
        return 1;
    }
}
