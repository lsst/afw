#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Image

#include "boost/test/unit_test.hpp"

#include <iostream>
#include "lsst/afw/image/Utils.h"

namespace image = lsst::afw::image;

BOOST_AUTO_TEST_CASE(bbox) {
    image::BBox bb;
    image::PointI point(1, 1);
    
    bb.grow(point);
    
    BOOST_CHECK(bb.contains(point));
}
