#include <iostream>
#include <sstream>
#include <ctime>

#include <lsst/fw/Image.h>

int main(int argc, char **argv) {
    typedef float imageType;
    const unsigned DefNIter = 100;

    if (argc < 2) {
        std::cout << "Usage: timePixelAccessor fitsFile [nIter]" << std::endl;
        std::cout << "fitsFile includes the \".fits\" suffix" << std::endl;
        std::cout << "nIter (default " << DefNIter << ") is the number of iterations" << std::endl;
        return 1;
    }
    
    unsigned nIter = DefNIter;
    if (argc > 2) {
        std::istringstream(argv[2]) >> nIter;
    }
    
    // read in fits files
    lsst::fw::Image<imageType> image;
    image.readFits(argv[1]);
    lsst::fw::Image<imageType>::pixel_accessor imOrigin = image.origin();
    
    unsigned imCols = image.getCols();
    unsigned imRows = image.getRows();
    
    std::cout << "Cols\tRows\tMPix\tSecPerIter\tSecPerIterPerMPix" << std::endl;
    
    clock_t startTime = clock();
    for (unsigned iter = 0; iter < nIter; ++iter) {
        lsst::fw::Image<imageType>::pixel_accessor imRow = imOrigin;
        for (unsigned row = 0; row < imRows; row++, imRow.next_row()) {
            lsst::fw::Image<imageType>::pixel_accessor imCol = imRow;
            for (unsigned col = 0; col < imCols; col++, imCol.next_col()) {
                *imCol += 1.0;
            }
        }
    }
    double secPerIter = (clock() - startTime) / static_cast<double> (nIter * CLOCKS_PER_SEC);
    double megaPix = static_cast<double>(imCols * imRows) / 1.0e6;
    double secPerMPixPerIter = secPerIter / static_cast<double>(megaPix);
    std::cout << imCols << "\t" << imRows << "\t" << megaPix << "\t" << secPerIter << "\t" << secPerMPixPerIter << std::endl;
}
