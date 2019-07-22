// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

/*
 * An example executible which calls the example 'stack' code
 */
#include <iostream>

#define BOOST_TEST_MODULE Stacker

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/included/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Stack.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

typedef image::Image<float> ImageF;
typedef image::MaskedImage<float> MImageF;
typedef std::vector<float> VecF;

BOOST_AUTO_TEST_CASE(
        MeanStack) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */

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
        if (iImg < nImg / 2) {
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
        knownWeightMean += wvec[iImg] * iImg;
        wsum += wvec[iImg];
    }
    knownMean /= nImg;
    knownWeightMean /= wsum;

    // ====================================================
    // regular image
    std::vector<std::shared_ptr<ImageF>> imgList;
    for (int iImg = 0; iImg < nImg; ++iImg) {
        std::shared_ptr<ImageF> img = std::shared_ptr<ImageF>(new ImageF(lsst::geom::Extent2I(nX, nY), iImg));
        imgList.push_back(img);
    }
    std::shared_ptr<ImageF> imgStack = math::statisticsStack<float>(imgList, math::MEAN);
    std::shared_ptr<ImageF> wimgStack = math::statisticsStack<float>(imgList, math::MEAN, sctrl, wvec);
    BOOST_CHECK_EQUAL((*imgStack)(nX / 2, nY / 2), knownMean);
    BOOST_CHECK_EQUAL((*wimgStack)(nX / 2, nY / 2), knownWeightMean);

    // ====================================================
    // masked image
    std::vector<std::shared_ptr<MImageF>> mimgList;
    for (int iImg = 0; iImg < nImg; ++iImg) {
        std::shared_ptr<MImageF> mimg = std::shared_ptr<MImageF>(new MImageF(lsst::geom::Extent2I(nX, nY)));
        *mimg->getImage() = iImg;
        *mimg->getMask() = 0x0;
        *mimg->getVariance() = iImg;
        mimgList.push_back(mimg);
    }
    std::shared_ptr<MImageF> mimgStack = math::statisticsStack<float>(mimgList, math::MEAN);
    std::shared_ptr<MImageF> wmimgStack = math::statisticsStack<float>(mimgList, math::MEAN, sctrl, wvec);
    BOOST_CHECK_EQUAL((*(mimgStack->getImage()))(nX / 2, nY / 2), knownMean);
    BOOST_CHECK_EQUAL((*(wmimgStack->getImage()))(nX / 2, nY / 2), knownWeightMean);

    // ====================================================
    // std::vector, and also with a constant weight vector
    std::vector<VecF> vecList;
    for (int iImg = 0; iImg < nImg; ++iImg) {
        VecF v(nX * nY, iImg);
        vecList.push_back(v);
    }
    VecF vecStack = math::statisticsStack<float>(vecList, math::MEAN);
    VecF wvecStack = math::statisticsStack<float>(vecList, math::MEAN, sctrl, wvec);
    BOOST_CHECK_EQUAL((vecStack)[nX * nY / 2], knownMean);
    BOOST_CHECK_EQUAL((wvecStack)[nX * nY / 2], knownWeightMean);
}
