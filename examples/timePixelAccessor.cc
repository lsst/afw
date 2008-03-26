#include <iostream>
#include <sstream>
#include <ctime>

#include <vw/Image.h>

int main(int argc, char **argv) {
    typedef float imageType;
    typedef vw::ImageView<imageType>::pixel_accessor accessorType;
    const unsigned DefNIter = 100;
    const unsigned DefNCols = 1024;

    if ((argc == 2) && (argv[1][0] == '-')) {
        std::cout << "Usage: timePixelAccessor [nIter [nCols [nRows]]]" << std::endl;
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
    
    vw::ImageView<imageType> image(nCols, nRows);
    accessorType imOrigin = image.origin();
    
    std::cout << "Cols\tRows\tMPix\tSecPerIter\tSecPerIterPerMPix" << std::endl;
    
    clock_t startTime = clock();
    for (unsigned iter = 0; iter < nIter; ++iter) {
        accessorType imRow = imOrigin;
        for (unsigned row = 0; row < nRows; row++, imRow.next_row()) {
            accessorType imCol = imRow;
            for (unsigned col = 0; col < nCols; col++, imCol.next_col()) {
                *imCol += 1.0;
            }
        }
    }
    double secPerIter = (clock() - startTime) / static_cast<double> (nIter * CLOCKS_PER_SEC);
    double megaPix = static_cast<double>(nCols * nRows) / 1.0e6;
    double secPerMPixPerIter = secPerIter / static_cast<double>(megaPix);
    std::cout << nCols << "\t" << nRows << "\t" << megaPix << "\t" << secPerIter << "\t\t" << secPerMPixPerIter << std::endl;
}
