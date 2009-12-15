// -*- lsst-c++ -*-
/**
 * @file simpleStacker.cc
 * @author Steve Bickerton
 * @brief An example executible which calls the example 'stack' code 
 *
 */
#include <iostream>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Stacker

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Stack.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

typedef image::Image<float> ImageF;

BOOST_AUTO_TEST_CASE(MeanStack) {
    
    int const nImg = 10;
    int const nX = 64;
    int const nY = 64;

    float knownMean = 0.0;
    std::vector<ImageF::Ptr> imgList;
    for (int iImg = 0; iImg < nImg; ++iImg) {
        ImageF::Ptr img = ImageF::Ptr (new ImageF(nX, nY, iImg));
        knownMean += iImg;
        imgList.push_back(img);
    }

    ImageF::Ptr imgStack = math::statisticsStack<float>(imgList, math::MEAN);
    knownMean /= nImg;

    BOOST_CHECK_EQUAL((*imgStack)(nX/2, nY/2), knownMean);
    
}
