// -*- LSST-C++ -*- // fixed format comment for emacs
/*
 * Test wcs copy constructor and assignment operator.
 */
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>

#include "lsst/daf/base.h"
#include "lsst/afw/image.h"

using lsst::daf::base::PropertySet;

int main() {
    typedef float Pixel;

    { //memory (de)allocation block
        char *afwdataCStr = getenv("AFWDATA_DIR");
        if (afwdataCStr == 0) {
            std::cout << "afwdata must be set up" << std::endl;
            exit(1);
        }
        std::string afwdata(afwdataCStr);
        const std::string inFilename(afwdata + "/small_MI");

        // Create a wcs from a fits file (so the wcs has some memory to allocate)
        std::cout << "Opening file " << inFilename << std::endl;

        PropertySet::Ptr metadata(new PropertySet);
        int const hdu = 0;
        lsst::afw::image::MaskedImage<Pixel, lsst::afw::image::MaskPixel> mImage(inFilename, hdu, metadata);
        lsst::afw::image::Wcs wcs(metadata);
        
        std::cout << "making a copy of a wcs" << std::endl;
        { // use copy constructor and deallocate the copy
            lsst::afw::image::Wcs wcsCopy(wcs);
            lsst::afw::image::Wcs wcsCopy2(wcsCopy);
        }
        std::cout << "deallocated the copy; assigning a wcs" << std::endl;
        { // use assignment operator and deallocate the assigned copy
            lsst::afw::image::Wcs wcsAssign, wcsAssign2;
            wcsAssign = wcs;
            wcsAssign2 = wcsAssign;
        }
        std::cout << "deallocated the assigned copy" << std::endl;
    } // close memory (de)allocation block

    // check for memory leaks
    if (lsst::daf::base::Citizen::census(0) != 0) {
        std::cerr << "Leaked memory blocks:" << std::endl;
        lsst::daf::base::Citizen::census(std::cerr);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
