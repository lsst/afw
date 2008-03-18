#include <iostream>
#include <sstream>
#include <ctime>

#include <lsst/fw/Image.h>

int main(int argc, char **argv) {
    typedef float imageType;
    const unsigned DefNIter = 100;

    if (argc < 3) {
        std::cout << "Usage: timeImageAddition fitsFile1 fitsFile2 [nIter]" << std::endl;
        std::cout << "fitsFile includes the \".fits\" suffix" << std::endl;
        std::cout << "nIter (default " << DefNIter << ") is the number of iterations" << std::endl;
        return 1;
    }
    
    unsigned nIter = DefNIter;
    if (argc > 3) {
        std::istringstream(argv[3]) >> nIter;
    }
    
    // read in fits files
    lsst::fw::Image<imageType> image1, image2;
    image1.readFits(argv[1]);
    image2.readFits(argv[2]);
    
    unsigned imCols = image1.getCols();
    unsigned imRows = image1.getRows();
    
    if ((imCols != image2.getCols()) || (imRows != image2.getRows())) {
        std::cerr << "Images must be the same size" << std::endl;
        std::cerr << argv[1] << " is " << imCols << "x" << imRows << std::endl;
        std::cerr << argv[2] << " is " << image2.getCols() << "x" << image2.getRows() << std::endl;
        return 1;
    }
    
    std::cout << "Cols\tRows\tMPix\tSecPerIter\tSecPerIterPerMPix" << std::endl;
    
    clock_t startTime = clock();
    for (unsigned iter = 0; iter < nIter; ++iter) {
        image1 += image2;
    }
    double secPerIter = (clock() - startTime) / static_cast<double> (nIter * CLOCKS_PER_SEC);
    double megaPix = static_cast<double>(imCols * imRows) / 1.0e6;
    double secPerMPixPerIter = secPerIter / static_cast<double>(megaPix);
    std::cout << imCols << "\t" << imRows << "\t" << megaPix << "\t" << secPerIter << "\t" << secPerMPixPerIter << std::endl;
}
