#include <cstdio>
#include <string>
#include <algorithm>

#include "lsst/afw/image.h"

namespace afwImage = lsst::afw::image;

int main() {
    afwImage::HealPixMapScheme hpScheme(2097152);
    afwImage::HealPixMapScheme::IdSet hpIdSet(hpScheme);
    afwImage::SkyMapImage<afwImage::HealPixId, double> hpPixelImage(hpScheme);
}
