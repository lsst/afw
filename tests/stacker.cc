// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
/**
 * @file simpleStacker.cc
 * @author Steve Bickerton
 * @brief An example executible which calls the example 'stack' code 
 *
 */
#include <iostream>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Stacker

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Stack.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;
namespace geom = lsst::afw::geom;

typedef image::Image<float> ImageF;
typedef image::MaskedImage<float> MImageF;
typedef std::vector<float> VecF;
typedef boost::shared_ptr<VecF> VecFPtr;

BOOST_AUTO_TEST_CASE(MeanStack) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    
    int const nImg = 10;
    int const nX = 64;
    int const nY = 64;

    // ===========================================================================
    // Plan: build lists (std::vectors) of Image, MaskedImage, and std::vector
    //       and set the pixels in each image to it's number in the list.
    // Crudely test the weighting by setting the weights to zero for the first half of the list

    
    // load a vector with weights to demonstrate weighting each image/vector by a constant weight.
    std::vector<float> wvec(nImg, 1.0);
    for (int iImg = 0; iImg < nImg; ++iImg) {
        if (iImg < nImg/2) {
            wvec[iImg] = 0.0;
        }
    }
    // we'll need a StatisticsControl object with weighted stats specified.
    math::StatisticsControl sctrl;
    sctrl.setWeighted(true);

    // get the known values
    float knownMean = 0.0;
    float knownWeightMean = 0.0;
    float wsum = 0.0;
    for (int iImg = 0; iImg < nImg; ++iImg) {
        knownMean += iImg;
        knownWeightMean += wvec[iImg]*iImg;
        wsum += wvec[iImg];
    }
    knownMean /= nImg;
    knownWeightMean /= wsum;

    
    // ====================================================
    // regular image
    std::vector<ImageF::Ptr> imgList;
    for (int iImg = 0; iImg < nImg; ++iImg) {
        ImageF::Ptr img = ImageF::Ptr (new ImageF(geom::Extent2I(nX, nY), iImg));
        imgList.push_back(img);
    }
    ImageF::Ptr imgStack = math::statisticsStack<float>(imgList, math::MEAN);
    ImageF::Ptr wimgStack = math::statisticsStack<float>(imgList, math::MEAN, sctrl, wvec);
    BOOST_CHECK_EQUAL((*imgStack)(nX/2, nY/2), knownMean);
    BOOST_CHECK_EQUAL((*wimgStack)(nX/2, nY/2), knownWeightMean);


    // ====================================================
    // masked image
    std::vector<MImageF::Ptr> mimgList;
    for (int iImg = 0; iImg < nImg; ++iImg) {
        MImageF::Ptr mimg = MImageF::Ptr(new MImageF(geom::Extent2I(nX, nY)));
        *mimg->getImage()    = iImg;
        *mimg->getMask()     = 0x0;
        *mimg->getVariance() = iImg;
        mimgList.push_back(mimg);
    }
    MImageF::Ptr mimgStack = math::statisticsStack<float>(mimgList, math::MEAN);
    MImageF::Ptr wmimgStack = math::statisticsStack<float>(mimgList, math::MEAN, sctrl, wvec);
    BOOST_CHECK_EQUAL((*(mimgStack->getImage()))(nX/2, nY/2), knownMean);
    BOOST_CHECK_EQUAL((*(wmimgStack->getImage()))(nX/2, nY/2), knownWeightMean);
    

    // ====================================================
    // std::vector, and also with a constant weight vector
    std::vector<VecFPtr> vecList;
    for (int iImg = 0; iImg < nImg; ++iImg) {
        VecFPtr v = VecFPtr(new VecF(nX*nY, iImg));
        vecList.push_back(v);
    }
    VecFPtr vecStack = math::statisticsStack<float>(vecList, math::MEAN);
    VecFPtr wvecStack = math::statisticsStack<float>(vecList, math::MEAN, sctrl, wvec);
    BOOST_CHECK_EQUAL((*vecStack)[nX*nY/2], knownMean);
    BOOST_CHECK_EQUAL((*wvecStack)[nX*nY/2], knownWeightMean);

}
