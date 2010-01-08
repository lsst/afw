#include <cstdio>
#include <string>
#include <algorithm>

#include "lsst/afw/image/Image.h"

namespace afwImage = lsst::afw::image;

template <typename PixelT>
void print(afwImage::Image<PixelT>& src, const std::string& title = "") {
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
        for (typename afwImage::Image<PixelT>::c_iterator src_it = src.row_begin(y); src_it != src.row_end(y);
            ++src_it) {
            printf("%4g ", static_cast<float>((*src_it)[0]));
        }
        printf("\n");
    }
}

/************************************************************************************************************/

int main(int argc, char *argv[]) {
    afwImage::DecoratedImage<float> dimg(10, 6);
    afwImage::Image<float> img(*dimg.getImage());

    std::string file_u16;
    if (argc == 2) {
        file_u16 = std::string(argv[1]);
    } else {
        std::string afwdata = getenv("AFWDATA_DIR");
        if (afwdata.empty()) {
            std::cerr << "AFWDATA_DIR not set.  Provide fits file as argument or setup afwdata.\n"
                      << std::endl;
            exit(EXIT_FAILURE);
        } else {
            file_u16 = afwdata + "/small_img.fits";
        }
    }
    std::cout << "Running with: " <<  file_u16 << std::endl;
    afwImage::DecoratedImage<float> dimg2(file_u16);

    return 0;
}
