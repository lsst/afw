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

int main() {
    typedef float pixelType;

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
        lsst::afw::image::MaskedImage<pixelType, lsst::afw::image::maskPixelType> mImage;
        mImage.readFits(inFilename);
        lsst::afw::image::Wcs wcs(mImage.getImage()->getMetaData());
        
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
    }
}
