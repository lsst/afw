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

    MaskedImage<ImagePixelType, MaskPixelType> testMasked;
    testMasked.readFits(name);

    DataProperty::PtrType metaDataPtr = testMasked.getImage()->getMetaData();

    Trace("MaskedImageIO_1", 1,
        boost::format("Number of FITS header cards: %d") 
            % FitsFormatter::countFITSHeaderCards(metaDataPtr, false));

    Trace("MaskedImageIO_1", 3,
        boost::format("FITS metadata string: %s") 
            % FitsFormatter::formatDataProperty(metaDataPtr, false));

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
