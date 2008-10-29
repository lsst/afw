#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Footprint

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/detection/Footprint.h"

namespace image = lsst::afw::image;
namespace detection = lsst::afw::detection;

typedef float ImagePixelT;

BOOST_AUTO_TEST_CASE(DetectionSets) {
    image::MaskedImage<ImagePixelT> img(10,20);
    *img.getImage() = 100;

    detection::DetectionSet<ImagePixelT> ds_by_value1(img, 0);
    BOOST_CHECK(ds_by_value1.getFootprints().size());

    detection::DetectionSet<ImagePixelT> ds_by_value2(img, detection::Threshold(0, detection::Threshold::VALUE));
    BOOST_CHECK(ds_by_value2.getFootprints().size());

    BOOST_CHECK_THROW(detection::DetectionSet<ImagePixelT>(img,         \
                                                           detection::Threshold(0, detection::Threshold::STDEV)), \
                      lsst::pex::exceptions::ExceptionStack);
    
    BOOST_CHECK_THROW(detection::DetectionSet<ImagePixelT>(img, \
                                                           detection::Threshold(0, detection::Threshold::VARIANCE)), \
                      lsst::pex::exceptions::ExceptionStack);
}
