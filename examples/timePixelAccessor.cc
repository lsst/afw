#include <iostream>
#include <sstream>
#include <ctime>

#include "lsst/afw/image.h"

namespace image = lsst::afw::image;

int main(int argc, char **argv) {
    typedef float imageType;
    typedef image::Image<imageType>::x_iterator iteratorType;

    int const DefNIter = 100;
    int const DefNCols = 1024;

    if ((argc == 2) && (argv[1][0] == '-')) {
        std::cout << "Usage: timePixelAccessor [nIter [nCols [nRows]]]" << std::endl;
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
    
    image::Image<imageType> image(nCols, nRows);

    std::cout << "Cols\tRows\tMPix\tSecPerIter\tSecPerIterPerMPix" << std::endl;
    
    clock_t startTime = clock();
    for (int iter = 0; iter < nIter; ++iter) {
        for (int y = 0; y != nRows; ++y) {
            for (iteratorType ptr = image.row_begin(y), end = image.row_end(y); ptr != end; ++ptr) {
                (*ptr)[0] += 1.0;
            }
        }
    }
    double secPerIter = (clock() - startTime) / static_cast<double> (nIter * CLOCKS_PER_SEC);
    double megaPix = static_cast<double>(nCols * nRows) / 1.0e6;
    double secPerMPixPerIter = secPerIter / static_cast<double>(megaPix);
    std::cout << nCols << "\t" << nRows << "\t" << megaPix << "\t" << secPerIter << "\t\t" << secPerMPixPerIter << std::endl;
}
