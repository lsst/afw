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
namespace geom = lsst::afw::geom;

typedef image::Image<float> ImageF;
typedef image::MaskedImage<float> MImageF;
typedef std::vector<float> VecF;
typedef std::shared_ptr<VecF> VecFPtr;

int main(int argc, char **argv) {

    int const nImg = 10;
    int const nX = 64;
    int const nY = 64;

    // load a vector with weights to demonstrate weighting each image/vector by a constant weight.
    std::vector<float> wvec(nImg, 1.0);
    for (int iImg = 0; iImg < nImg; ++iImg) {
        if (iImg < nImg/2) {
            wvec[iImg] = 0.0;
        }
    }
    // we'll need a StatisticsControl object with weighted stats specified.
    math::StatisticsControl sctrl;
    if ( argc > 1 && std::atoi(argv[1]) > 0 ) {
        sctrl.setWeighted(true);
    } else {
        sctrl.setWeighted(false);
    }

    // regular image
    std::vector<ImageF::Ptr> imgList;
    for (int iImg = 0; iImg < nImg; ++iImg) {
        ImageF::Ptr img = ImageF::Ptr (new ImageF(geom::Extent2I(nX, nY), iImg));
        imgList.push_back(img);
    }
    ImageF::Ptr imgStack = math::statisticsStack<float>(imgList, math::MEAN);
    std::cout << "Image:                      " << (*imgStack)(nX/2, nY/2) << std::endl;
    ImageF::Ptr wimgStack = math::statisticsStack<float>(imgList, math::MEAN, sctrl, wvec);
    std::cout << "Image (const weight):       " << (*wimgStack)(nX/2, nY/2) << std::endl;


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
    std::cout << "MaskedImage:                " << (*mimgStack->getImage())(nX/2, nY/2) << std::endl;
    MImageF::Ptr wmimgStack = math::statisticsStack<float>(mimgList, math::MEAN, sctrl, wvec);
    std::cout << "MaskedImage (const weight): " << (*wmimgStack->getImage())(nX/2, nY/2) << std::endl;



    // std::vector, and also with a constant weight vector
    std::vector<VecFPtr> vecList;
    for (int iImg = 0; iImg < nImg; ++iImg) {
        VecFPtr v = VecFPtr(new VecF(nX*nY, iImg));
        vecList.push_back(v);
    }
    VecFPtr vecStack = math::statisticsStack<float>(vecList, math::MEAN);
    std::cout << "Vector:                     " << (*vecStack)[nX*nY/2] << std::endl;
    VecFPtr wvecStack = math::statisticsStack<float>(vecList, math::MEAN, sctrl, wvec);
    std::cout << "Vector (const weight):      " << (*wvecStack)[nX*nY/2] << std::endl;

}
