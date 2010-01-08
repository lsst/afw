#include <cstdio>
#include <string>
#include <algorithm>

#include "lsst/afw/image/Mask.h"
#include "lsst/afw/image/LsstImageTypes.h"

namespace afwImage = lsst::afw::image;

/************************************************************************************************************/

int main() {
    afwImage::Mask<afwImage::MaskPixel> img(10, 6);
    // This is equivalent to mask = 100:
    for (afwImage::Mask<afwImage::MaskPixel>::iterator ptr = img.begin(); ptr != img.end(); ++ptr) {
        (*ptr)[0] = 100;
    }
    // so is this, but fills backwards
    for (afwImage::Mask<afwImage::MaskPixel>::reverse_iterator ptr = img.rbegin(); ptr != img.rend(); ++ptr) {
        (*ptr)[0] = 100;
    }
    // so is this, but tests a different way of choosing begin()
    for (afwImage::Mask<afwImage::MaskPixel>::iterator ptr = img.at(0, 0); ptr != img.end(); ++ptr) {
        (*ptr)[0] = 100;
    }

    afwImage::Mask<afwImage::MaskPixel> jmg = img;

    printf("%dx%d\n", img.getWidth(), img.getHeight());

    *img.y_at(7, 2) = 999;
    *img.x_at(0, 0) = 0;
    img(img.getWidth() - 1, img.getHeight() - 1) = -100;

    printf("sub Mask<afwImage::MaskPixel>s\n");
#if 0
    // img will be modified
    afwImage::Mask<afwImage::MaskPixel> simg(img, afwImage::BBox(afwImage::PointI(1, 1), 5, 2));
#elif 0
    // img will not be modified
    afwImage::Mask<afwImage::MaskPixel> simg(img, afwImage::BBox(afwImage::PointI(1, 1), 5, 2), true);
#else
    // img will be modified
    afwImage::Mask<afwImage::MaskPixel> simg1(img, afwImage::BBox(afwImage::PointI(1, 1), 7, 3));
    afwImage::Mask<afwImage::MaskPixel> simg(simg1, afwImage::BBox(afwImage::PointI(0, 0), 5, 2));
#endif

#if 0
    simg = 0;
#elif 1
    {
        afwImage::Mask<afwImage::MaskPixel> nimg(5, 2);
        nimg = 1;
        simg <<= nimg;
    }
#endif    

    for (int r = 0; r != img.getHeight(); ++r) {
        std::fill(img.row_begin(r), img.row_end(r), 100*(1 + r));
    }

    
    std::string afwdata(getenv("AFWDATA_DIR"));
    std::string smallMaskFile;
    if (afwdata.empty()) {
        std::cerr << "AFWDATA_DIR not set." << std::endl;
        exit(EXIT_FAILURE);
    } else {
        smallMaskFile = afwdata + "/small_MI_msk.fits";
    }

    
    afwImage::Mask<afwImage::MaskPixel> msk(smallMaskFile);
    printf("msk(0,0) = %d\n", msk(0,0));
    
    afwImage::DecoratedImage<unsigned short> dimg(smallMaskFile);
    //Image<unsigned short>::Ptr img = dimg.getImage();
    printf("dimg(0,0) = %d\n", (*(dimg.getImage()))(0,0));

    return 0;
}
