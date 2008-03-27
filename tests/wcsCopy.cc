// -*- LSST-C++ -*- // fixed format comment for emacs
/*
 * Test wcs copy constructor and assignment operator.
 */
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>

#include <lsst/daf/data.h>
#include <lsst/afw/image.h>

int main() {
    typedef float pixelType;

    { //memory (de)allocation block
        char *fwDataCStr = getenv("FWDATA_DIR");
        if (fwDataCStr == 0) {
            std::cout << "fwData must be set up" << std::endl;
            exit(1);
        }
        std::string fwData(fwDataCStr);
        const std::string inFilename(fwData + "/small_MI");

        // Create a wcs from a fits file (so the wcs has some memory to allocate)
        std::cout << "Opening file " << inFilename << std::endl;
        lsst::afw::image::MaskedImage<pixelType, lsst::afw::image::maskPixelType> mImage;
        mImage.readFits(inFilename);
        lsst::afw::image::WCS wcs(mImage.getImage()->getMetaData());
        
        std::cout << "making a copy of a wcs" << std::endl;
        { // use copy constructor and deallocate the copy
            lsst::afw::image::WCS wcsCopy(wcs);
            lsst::afw::image::WCS wcsCopy2(wcsCopy);
        }
        std::cout << "deallocated the copy; assigning a wcs" << std::endl;
        { // use assignment operator and deallocate the assigned copy
            lsst::afw::image::WCS wcsAssign, wcsAssign2;
            wcsAssign = wcs;
            wcsAssign2 = wcsAssign;
        }
        std::cout << "deallocated the assigned copy" << std::endl;
    } // close memory (de)allocation block

    // check for memory leaks
    if (lsst::daf::data::Citizen::census(0) != 0) {
        std::cerr << "Leaked memory blocks:" << std::endl;
        lsst::daf::data::Citizen::census(std::cerr);
    }
}
