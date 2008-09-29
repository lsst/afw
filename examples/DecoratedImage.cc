#include <cstdio>
#include <string>
#include <algorithm>
#include "lsst/afw/image/Image.h"

using namespace lsst::afw::image;

template <typename PixelT>
void print(Image<PixelT>& src, const std::string& title = "") {
    if (title.size() > 0) {
        printf("%s:\n", title.c_str());
    }

    printf("%3s ", "");
    for (int x = 0; x != src.getWidth(); ++x) {
        printf("%4d ", x);
    }
    printf("\n");

    for (int y = src.getHeight() - 1; y >= 0; --y) {
        printf("%3d ", y);
        for (typename Image<PixelT>::c_iterator src_it = src.row_begin(y); src_it != src.row_end(y); ++src_it) {
            printf("%4g ", static_cast<float>((*src_it)[0]));
        }
        printf("\n");
    }
}

/************************************************************************************************************/

int main() {
    DecoratedImage<float> dimg(10, 6);
    Image<float> img(*dimg.getImage());

    const char *file_u16 = "/u/rhl/LSST/gil/src/Fits/small.fits";
    DecoratedImage<float> dimg2(file_u16);

    return 0;
}
