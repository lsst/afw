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
typedef image::MaskedImage<float> MImageF;
typedef std::vector<float> VecF;
typedef boost::shared_ptr<VecF> VecFPtr;

int main (int argc, char **argv) {

    int const nImg = 10;
    int const nX = 64;
    int const nY = 64;

    // regular image
    std::vector<ImageF::Ptr> imgList;
    for (int iImg = 0; iImg < nImg; ++iImg) {
        ImageF::Ptr img = ImageF::Ptr (new ImageF(nX, nY, iImg));
        imgList.push_back(img);
    }
    ImageF::Ptr imgStacker = math::statisticsStack<float>(imgList, math::MEAN);
    std::cout << (*imgStacker)(nX/2, nY/2) << std::endl;


    // masked image
    std::vector<MImageF::Ptr> mimgList;
    for (int iImg = 0; iImg < nImg; ++iImg) {
        MImageF::Ptr mimg = MImageF::Ptr(new MImageF(nX,nY));
        *mimg->getImage()    = iImg;
        *mimg->getMask()     = 0x0;
        *mimg->getVariance() = iImg;
        mimgList.push_back(mimg);
    }
    MImageF::Ptr mimgStacker = math::statisticsStack<float>(mimgList, math::MEAN);
    std::cout << (*mimgStacker->getImage())(nX/2, nY/2) << std::endl;
    


    // std::vector
    std::vector<VecFPtr> vecList;
    for (int iImg = 0; iImg < nImg; ++iImg) {
        VecFPtr v = VecFPtr(new VecF(nX*nY, iImg));
        vecList.push_back(v);
    }
    VecFPtr vecStacker = math::statisticsStack<float>(vecList, math::MEAN);
    std::cout << (*vecStacker)[nX*nY/2] << std::endl;

    
}
