// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>
#include <sstream>
#include <ctime>

#include "lsst/afw/image.h"

namespace image = lsst::afw::image;
namespace geom =lsst::afw::geom;

int main(int argc, char **argv) {
    typedef float imageType;
    const unsigned DefNIter = 100;
    const unsigned DefNCols = 1024;
    
    if ((argc == 2) && (argv[1][0] == '-')) {
        std::cout << "Usage: timeImageAddition [nIter [nCols [nRows]]]" << std::endl;
        std::cout << "nIter (default " << DefNIter << ") is the number of iterations" << std::endl;
        std::cout << "nCols (default " << DefNCols << ") is the number of columns" << std::endl;
        std::cout << "nRows (default = nCols) is the number of rows" << std::endl;
        return 1;
    }
    
    unsigned nIter = DefNIter;
    if (argc > 1) {
        std::istringstream(argv[1]) >> nIter;
    }
    unsigned nCols = DefNCols;
    if (argc > 2) {
        std::istringstream(argv[2]) >> nCols;
    }
    unsigned nRows = nCols;
    if (argc > 3) {
        std::istringstream(argv[3]) >> nRows;
    }
    
    image::Image<imageType> image1(geom::Extent2I(nCols, nRows));
    image::Image<imageType> image2(image1.getDimensions());
    
    std::cout << "Cols\tRows\tMPix\tSecPerIter\tSecPerIterPerMPix" << std::endl;
    
    clock_t startTime = clock();
    for (unsigned iter = 0; iter < nIter; ++iter) {
        image1 += image2;
    }
    double secPerIter = (clock() - startTime) / static_cast<double> (nIter * CLOCKS_PER_SEC);
    double megaPix = static_cast<double>(nCols * nRows) / 1.0e6;
    double secPerMPixPerIter = secPerIter / static_cast<double>(megaPix);
    std::cout << nCols << "\t" << nRows << "\t" << megaPix << "\t" << secPerIter << "\t\t" <<
        secPerMPixPerIter << std::endl;
}
