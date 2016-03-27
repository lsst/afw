/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>
#include <sstream>
#include <ctime>

#include "boost/format.hpp"
#include "boost/gil/gil_all.hpp"

template<class ImageT>
void timePixelAccess(ImageT const &image, float const pix, int nIter) {
    const int nCols = image.width();
    const int nRows = image.height();

    clock_t startTime = clock();
    for (int iter = 0; iter < nIter; ++iter) {
        for (int y = 0; y < image.height(); ++y) {
            for (typename ImageT::x_iterator ptr = image.row_begin(y), end = image.row_end(y);
                ptr != end; ++ptr) {
                (*ptr)[0] += pix;
            }
        }
    }
    double secPerIter = (clock() - startTime) / static_cast<double> (nIter * CLOCKS_PER_SEC);
    double const megaPix = static_cast<double>(nCols * nRows) / 1.0e6;
    std::cout << boost::format("Pixel Iterator\t%d\t%d\t%g\t%-8g\t%-8.1f") %
        nCols % nRows % megaPix % secPerIter % (megaPix/secPerIter) << std::endl;

    startTime = clock();
    for (int iter = 0; iter < nIter; ++iter) {
        for (int y = 0; y < image.height(); ++y) {
            for (typename ImageT::xy_locator ptr = image.xy_at(0, y), end = image.xy_at(nCols, y);
                ptr != end; ++ptr.x()) {
                (*ptr)[0] += pix;
            }
        }
    }
    secPerIter = (clock() - startTime) / static_cast<double> (nIter * CLOCKS_PER_SEC);
    std::cout << boost::format("Pixel Locator\t%d\t%d\t%g\t%-8g\t%-8.1f") %
        nCols % nRows % megaPix % secPerIter % (megaPix/secPerIter) << std::endl;
}

int main(int argc, char **argv) {
    int const DefNIter = 100;
    int const DefNCols = 1024;

    if ((argc == 2) && (argv[1][0] == '-')) {
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
      
    std::cout << "Accessor Type\tCols\tRows\tMPix\tSecPerIter\tSecPerIterPerMPix" << std::endl;
        std::cout << "Image(" << nCols << ", " << nRows << ")" << std::endl;

    boost::gil::gray32f_image_t image(nCols, nRows);

    float pix = 1.0;
    timePixelAccess(boost::gil::view(image), pix, nIter);
}
