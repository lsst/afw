#include <iostream>
#include <vector>
/**
* WARNING: CONTAINS HARD-CODED PATHS (TEMPORARILY FOR DEBUGGING)
*/

#include "boost/format.hpp"

#include "lsst/afw/math.h"

namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;

int main() {
    typedef double ImageType;
    afwMath::LanczosWarpingKernel<afwImage::MaskedImage<ImageType>, afwImage::MaskedImage<ImageType> > warpingKernel(3);
    
    afwImage::Exposure<ImageType> srcExposure("/Users/rowen/LSST/code/testdata/afwdata/small_MI");
    afwImage::Exposure<ImageType> destExposure("/Users/rowen/LSST/code/testdata/afwdata/small_MI");
    
    int nGood = afwMath::warpExposure(destExposure, srcExposure, warpingKernel);
    std::cout << nGood << " pixels\n";
}
