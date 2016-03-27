/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>
#include <sstream>
#include <ctime>

#include "boost/format.hpp"

#include "lsst/afw/image.h"

namespace afwImage = lsst::afw::image;
namespace geom = lsst::afw::geom;

template<class ImageT>
void timePixelAccess(ImageT const &image, typename ImageT::SinglePixel const pix, int nIter) {
    const int nCols = image.getWidth();
    const int nRows = image.getHeight();

    clock_t startTime = clock();
    for (int iter = 0; iter < nIter; ++iter) {
        for (int y = 0; y < image.getHeight(); ++y) {
            for (typename ImageT::x_iterator ptr = image.row_begin(y), end = image.row_end(y);
                ptr != end; ++ptr) {
                *ptr += pix;
            }
        }
    }
    double secPerIter = (clock() - startTime) / static_cast<double> (nIter * CLOCKS_PER_SEC);
    double const megaPix = static_cast<double>(nCols * nRows) / 1.0e6;
    std::cout << boost::format("Pixel Iterator\t%d\t%d\t%g\t%-8g\t%-8.1f") %
        nCols % nRows % megaPix % secPerIter % (megaPix/secPerIter) << std::endl;

    startTime = clock();
    for (int iter = 0; iter < nIter; ++iter) {
        for (int y = 0; y < image.getHeight(); ++y) {
            for (typename ImageT::xy_locator ptr = image.xy_at(0, y), end = image.xy_at(nCols, y);
                ptr != end; ++ptr.x()) {
                *ptr += pix;
            }
        }
    }
    secPerIter = (clock() - startTime) / static_cast<double> (nIter * CLOCKS_PER_SEC);
    std::cout << boost::format("Pixel Locator\t%d\t%d\t%g\t%-8g\t%-8.1f") %
        nCols % nRows % megaPix % secPerIter % (megaPix/secPerIter) << std::endl;
}

int main(int argc, char **argv) {
    typedef float ImagePixel;

    int const DefNIter = 100;
    int const DefNCols = 1024;

    if ((argc >= 2) && (argv[1][0] == '-')) {
        std::cout << "Usage: timePixelAccess [nIter [nCols [nRows]]]" << std::endl;
        std::cout << "nIter (default " << DefNIter << ") is the number of iterations" << std::endl;
        std::cout << "nCols (default " << DefNCols << ") is the number of columns" << std::endl;
        std::cout << "nRows (default = nCols) is the number of rows" << std::endl;
        return 1;
    }
    
    int nIter = DefNIter;
    if (argc > 1) {
        std::istringstream(argv[1]) >> nIter;
    }
    int nCols = DefNCols;
    if (argc > 2) {
        std::istringstream(argv[2]) >> nCols;
    }
    int nRows = nCols;
    if (argc > 3) {
        std::istringstream(argv[3]) >> nRows;
    }
    
    std::cout << "Accessor Type\tCols\tRows\tMPix\tSecPerIter\tMPixPerSec" << std::endl;

    std::cout << "Image(" << nCols << ", " << nRows << ")" << std::endl;
    {
        afwImage::Image<ImagePixel> image(geom::Extent2I(nCols, nRows));
        afwImage::Image<ImagePixel>::SinglePixel pix(1.0);
        timePixelAccess(image, pix, nIter);
    }
    
    std::cout << "MaskedImage(" << nCols << ", " << nRows << ")" << std::endl;
    {
        afwImage::MaskedImage<ImagePixel> maskedImage(geom::Extent2I(nCols, nRows));
        afwImage::MaskedImage<ImagePixel>::SinglePixel pix(1.0, 0x10, 100);
        timePixelAccess(maskedImage, pix, nIter);
    }
}
