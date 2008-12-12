#include <iostream>
#include <cmath>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Background

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Background.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

typedef image::Image<float> ImageT;

BOOST_AUTO_TEST_CASE(Background) {

    int nx = 10;
    int ny = 40;
    ImageT img(nx, ny);
    ImageT::Pixel const pixval = 10000;
    img = pixval;

    {
        int xcen = nx/2;
        int ycen = ny/2;
        math::BackgroundControl bgCtrl;
        math::Background<ImageT> back = math::make_Background(img, bgCtrl);
        double const testval = back.getPixel(xcen, ycen);
        
        BOOST_CHECK_EQUAL(testval, pixval);

    }

}
