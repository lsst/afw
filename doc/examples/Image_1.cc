/// \file
#include "lsst/afw/image/Image.h"

namespace image = lsst::afw::image;

int main() {
    typedef int PixelT;
    image::Image<PixelT> img(10, 6);
    img = 0;
    
    // This is equivalent to img = 100:
    for (image::Image<PixelT>::iterator ptr = img.begin(); ptr != img.end(); ++ptr) {
        *ptr = 100;
    }
    // so is this, but fills backwards (and only finds end once)
    for (image::Image<PixelT>::reverse_iterator ptr = img.rbegin(), rend = img.rend(); ptr != rend; ++ptr) {
        *ptr = 100;
    }
    // so is this, but shows a different way of choosing begin()
    for (image::Image<PixelT>::iterator ptr = img.at(0, 0); ptr != img.end(); ++ptr) {
        *ptr = 100;
    }

    // Set the pixels row by row, to avoid repeated checks for end-of-row
    for (int y = 0; y != img.getHeight(); ++y) {
        for (image::Image<PixelT>::x_iterator ptr = img.row_begin(y); ptr != img.row_end(y); ++ptr) {
            *ptr = 100;
        }
    }
    // Set the pixels column by column, with awful consequences upon cache performance
    for (int x = 0; x != img.getWidth(); ++x) {
        for (image::Image<PixelT>::y_iterator ptr = img.col_begin(x), end = img.col_end(x); ptr != end; ++ptr) {
            *ptr = 100;
        }
    }
    // Set the pixels column by column in batches to avoid some of the worst effects upon cache performance
    int x = 0;
    for (; x != img.getWidth()%4; ++x) {
        for (image::Image<PixelT>::y_iterator ptr = img.col_begin(x), end = img.col_end(x); ptr != end; ++ptr) {
            *ptr = 100;
        }
    }
    for (; x != img.getWidth(); x += 4) {
        for (image::Image<PixelT>::y_iterator ptr0 = img.col_begin(x+0), end0 = img.col_end(x+0),
                                              ptr1 = img.col_begin(x+1), end1 = img.col_end(x+1),
                                              ptr2 = img.col_begin(x+2), end2 = img.col_end(x+2),
                                              ptr3 = img.col_begin(x+3), end3 = img.col_end(x+3);
             ptr0 != end0; ++ptr0, ++ptr1, ++ptr2, ++ptr3) {
            *ptr0 = *ptr1 = *ptr2 = *ptr3 = 100;
        }
    }

    return 0;
}
