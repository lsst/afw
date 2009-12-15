// -*- lsst-c++ -*-
/**
 * @file simpleStacker.cc
 * @author Steve Bickerton
 * @brief An example executible which calls the example 'stack' code 
 *
 */
#include <iostream>

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Stack.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

typedef image::Image<float> ImageF;

int main (int argc, char **argv) {

    int const nImg = 10;
    int const nX = 64;
    int const nY = 64;
    
    std::vector<ImageF::Ptr> imgList;
    for (int iImg = 0; iImg < nImg; ++iImg) {
        ImageF::Ptr img = ImageF::Ptr (new ImageF(nX, nY, iImg));
        imgList.push_back(img);
    }

    ImageF::Ptr imgStacker = math::statisticsStack<float>(imgList, math::MEAN);
    
    std::cout << (*imgStacker)(nX/2, nY/2) << std::endl;
    
}
