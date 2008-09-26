#include <cstdio>
#include <string>
#include <algorithm>
#include "lsst/gil/Mask.h"

using namespace lsst::afw::image;

/************************************************************************************************************/

int main() {
#if 1
    Mask<MaskPixel> img(10, 6);
    // This is equivalent to mask = 100:
    for (Mask<MaskPixel>::iterator ptr = img.begin(); ptr != img.end(); ++ptr) {
        (*ptr)[0] = 100;
    }
    // so is this, but fills backwards
    for (Mask<MaskPixel>::reverse_iterator ptr = img.rbegin(); ptr != img.rend(); ++ptr) {
        (*ptr)[0] = 100;
    }
    // so is this, but tests a different way of choosing begin()
    for (Mask<MaskPixel>::iterator ptr = img.at(0, 0); ptr != img.end(); ++ptr) {
        (*ptr)[0] = 100;
    }

    Mask<MaskPixel> jmg = img;

    printf("%dx%d\n", img.getWidth(), img.getHeight());

    *img.y_at(7, 2) = 999;
    *img.x_at(0, 0) = 0;
    img(img.getWidth() - 1, img.getHeight() - 1) = -100;

    printf("sub Mask<MaskPixel>s\n");
#if 0
    Mask<MaskPixel> simg = Mask<MaskPixel>(img, Bbox(PointI(1, 1), 5, 2)); // img will be modified
#elif 0
    Mask<MaskPixel> simg = Mask<MaskPixel>(img, Bbox(PointI(1, 1), 5, 2), true); // img won't be modified
#else
    Mask<MaskPixel> simg1 = Mask<MaskPixel>(img, Bbox(PointI(1, 1), 7, 3)); // img will be modified
    Mask<MaskPixel> simg = Mask<MaskPixel>(simg1, Bbox(PointI(0, 0), 5, 2));
#endif

#if 0
    simg = 0;
#elif 1
    {
        Mask<MaskPixel> nimg = Mask<MaskPixel>(5, 2);
        nimg = 1;
        simg <<= nimg;
    }
#endif    

    for (int r = 0; r != img.getHeight(); ++r) {
        std::fill(img.row_begin(r), img.row_end(r), 100*(1 + r));
    }
#else
    Mask<MaskPixel> msk("/u/rhl/LSST/DMS/afwdata/small_MI_msk.fits");
    printf("msk(0,0) = %d\n", msk(0,0)[0]);
    
    DecoratedImage<unsigned short> dimg("/u/rhl/LSST/DMS/afwdata/small_MI_msk.fits");
    //Image<unsigned short>::Ptr img = dimg.getImage();
    printf("dimg(0,0) = %d\n", (*(dimg.getImage()))(0,0)[0]);
#endif

    return 0;
}
