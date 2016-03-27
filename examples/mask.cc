/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <cstdio>
#include <string>
#include <algorithm>

#include "lsst/utils/Utils.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;

/************************************************************************************************************/

int main() {
    afwImage::Mask<afwImage::MaskPixel> img(afwGeom::Extent2I(10, 6));
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
    img(img.getWidth() - 1, img.getHeight() - 1) = 100;

    printf("sub Mask<afwImage::MaskPixel>s\n");

    // img will be modified
    afwImage::Mask<afwImage::MaskPixel> simg1(
        img, 
        afwGeom::Box2I(afwGeom::Point2I(1, 1), afwGeom::Extent2I(7, 3)),
        afwImage::LOCAL
    );
    afwImage::Mask<afwImage::MaskPixel> simg(
        simg1, 
        afwGeom::Box2I(afwGeom::Point2I(0, 0), afwGeom::Extent2I(5, 2)),
        afwImage::LOCAL
    );

    {
        afwImage::Mask<afwImage::MaskPixel> nimg(simg.getDimensions());
        nimg = 1;
        simg.assign(nimg);
    }

    for (int r = 0; r != img.getHeight(); ++r) {
        std::fill(img.row_begin(r), img.row_end(r), 100*(1 + r));
    }

    std::string inImagePath;
    try {
        std::string dataDir = lsst::utils::getPackageDir("afwdata");
        inImagePath = dataDir + "/data/small.fits";
    } catch (lsst::pex::exceptions::NotFoundError) {
        std::cerr << "Usage: mask [fitsFile]" << std::endl;
        std::cerr << "fitsFile is the path to a masked image" << std::endl;
        std::cerr << "\nError: setup afwdata or specify fitsFile.\n" << std::endl;
        exit(EXIT_FAILURE);
    }

    afwImage::MaskedImage<float> mi = afwImage::MaskedImage<float>(inImagePath);
    printf("mask(0,0) = %d\n", (*(mi.getMask()))(0,0));
    printf("image(0,0) = %f\n", (*(mi.getImage()))(0,0));

    return 0;
}
