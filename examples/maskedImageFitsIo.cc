/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>
#include <sstream>

#include "lsst/utils/Utils.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/MaskedImage.h"

const std::string outImagePath("rwfitsOut.fits");

int main(int argc, char **argv) {

    std::string inImagePath;
    if (argc == 2) {
        inImagePath = std::string(argv[1]);
    } else {
        try {
            std::string dataDir = lsst::utils::getPackageDir("afwdata");
            inImagePath = dataDir + "/data/small.fits";
        } catch (lsst::pex::exceptions::NotFoundError) {
            std::cerr << "Usage: maskedImageFitsIO [fitsFile]" << std::endl;
            std::cerr << "fitsFile is the path to a masked image" << std::endl;
            std::cerr << "\nError: setup afwdata or specify fitsFile.\n" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Running with: " <<  inImagePath << std::endl;

    lsst::afw::image::MaskedImage<float> mImage(inImagePath);
    mImage.writeFits(outImagePath);
    std::cout << "Wrote masked image: " << outImagePath << std::endl;
}
