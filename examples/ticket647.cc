/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"

namespace image = lsst::afw::image;
namespace geom = lsst::afw::geom;

int main() {
    image::MaskedImage<int> mi(geom::Extent2I(10,10));
    image::Image<int>       im(mi.getDimensions());

    image::MaskedImage<int>::xy_locator mi_loc = mi.xy_at(5,5);
    image::Image<int>::xy_locator       im_loc = im.xy_at(5,5);

    std::pair<int, int> const step = std::make_pair(1,1);

    mi_loc += step;
    im_loc += step;
    
    return 0;
}
