#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Image

#include "boost/test/unit_test.hpp"

#include <iostream>
#include "lsst/afw/image/Utils.h"

namespace image = lsst::afw::image;

BOOST_AUTO_TEST_CASE(bbox) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    image::BBox bb;
    image::PointI point(1, 1);
    
    bb.grow(point);
    
    BOOST_CHECK(bb.contains(point));
}
