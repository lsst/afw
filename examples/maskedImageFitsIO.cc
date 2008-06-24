#include <iostream>
#include <sstream>

#include "lsst/afw/image/MaskedImage.h"

const std::string outFile("rwfitsOut");

int main(int argc, char **argv) {
    typedef float pixelType;

    lsst::afw::image::MaskedImage<pixelType, lsst::afw::image::maskPixelType> mImage;

    if (argc < 2) {
        std::cerr << "Usage: maskedImageFitsIO fitsFile" << std::endl;
        std::cerr << "fitsFile excludes the \"_img.fits\" suffix" << std::endl;
        return 1;
    }

    mImage.readFits(argv[1]);
    mImage.writeFits(outFile);
}
