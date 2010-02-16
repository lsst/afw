// -*- lsst-c++ -*-
#include <stdexcept>

#include "lsst/daf/base.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image.h"
#include "lsst/afw/math.h"
#include "lsst/afw/formatters/Utils.h"

using namespace std;

using lsst::pex::logging::Trace;
using lsst::daf::base::PropertySet;


namespace pexEx = lsst::pex::exceptions;

/*
 * Make this a subroutine so that locals go out of scope as part of test
 * of memory management
 */
void test(char *name) {

    typedef lsst::afw::image::MaskPixel MaskPixelType;
    typedef float ImagePixelType;

    PropertySet::Ptr metadata(new PropertySet());
    int const hdu = 0;
    lsst::afw::image::MaskedImage<ImagePixelType, MaskPixelType> testMasked(name, hdu, metadata);

    Trace("MaskedImageIO_1", 1,
          boost::format("Number of FITS header cards: %d") 
          % lsst::afw::formatters::countFitsHeaderCards(metadata));

    Trace("MaskedImageIO_1", 3,
        boost::format("FITS metadata string: %s") 
            % lsst::afw::formatters::formatFitsProperties(metadata));

    lsst::afw::image::Wcs testWcs(metadata);

    lsst::afw::image::PointD pix, sky;

//     pix[0] = testMasked.getCols() / 2.0;
//     pix[1] = testMasked.getRows() / 2.0;
    pix[0] = 200;
    pix[1] = 180;

    sky = testWcs.pixelToSky(pix);

    Trace("MaskedImageIO_1", 1,
          boost::format("pix: %lf %lf") % pix[0] % pix[1]);

    Trace("MaskedImageIO_1", 1,
          boost::format("sky: %lf %lf") % sky[0] % sky[1]);

    sky = testWcs.skyToPixel(pix);

    Trace("MaskedImageIO_1", 1,
          boost::format("pix: %lf %lf") % pix[0] % pix[1]);


}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: inputBaseName" << endl;
        return EXIT_FAILURE;
    }

    int status = EXIT_SUCCESS; 
    Trace::setDestination(cout);
    Trace::setVerbosity(".", 1);

    try {
        test(argv[1]);
    } catch (pexEx::Exception &e) {
        clog << e.what() << endl;
        status = EXIT_FAILURE;
    }
    // Check for memory leaks
    if (lsst::daf::base::Citizen::census(0) == 0) {
        cerr << "No leaks detected" << endl;
    } else {
        cerr << "Leaked memory blocks:" << endl;
        lsst::daf::base::Citizen::census(cerr);
        status = EXIT_FAILURE;
    }
    return status;
}

