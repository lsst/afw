#include <iostream>
#include <sstream>

#include "lsst/afw/image/MaskedImage.h"

const std::string outFile("rwfitsOut");

int main(int argc, char **argv) {

    std::string file;
    if (argc == 2) {
        file = std::string(argv[1]);
    } else {
        std::string afwdata = getenv("AFWDATA_DIR");
        if (afwdata.empty()) {
            std::cerr << "Usage: maskedImageFitsIO fitsFile" << std::endl;
            std::cerr << "fitsFile excludes the \"_img.fits\" suffix" << std::endl;
            std::cerr << "AFWDATA_DIR not set.  Provide fits file as argument or setup afwdata.\n"
                      << std::endl;
            exit(EXIT_FAILURE);
        } else {
            file = afwdata + "/small_MI";
        }
    }
    std::cout << "Running with: " <<  file << std::endl;

    lsst::afw::image::MaskedImage<float> mImage(file);
    mImage.writeFits(outFile);
}
