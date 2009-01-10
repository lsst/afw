#include <iostream>
#include <string>
#include <algorithm>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE DecoratedImage

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/Image.h"

namespace daf_base = lsst::daf::base;
namespace image = lsst::afw::image;
using namespace std;

typedef float PixelT;
typedef image::Image<PixelT> ImageT;
typedef image::DecoratedImage<PixelT> DecoratedImageT;

/************************************************************************************************************/

DecoratedImageT make_image(int const width=5, int const height=6) {
    DecoratedImageT dimg(width, height);
    ImageT::Ptr img = dimg.getImage();

    int i = 0;
    for (ImageT::iterator ptr = img->begin(), end = img->end(); ptr != end; ++ptr, ++i) {
        *ptr = i/dimg.getWidth() + 100*(i%dimg.getWidth());
    }

    return dimg;
}

/************************************************************************************************************/

BOOST_AUTO_TEST_CASE(setValues) {
    DecoratedImageT dimg = make_image();
    daf_base::PropertySet::Ptr metadata = dimg.getMetadata();

    metadata->add("RHL", 1);
}
