// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Footprint

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/detection.h"

namespace image = lsst::afw::image;
namespace detection = lsst::afw::detection;
namespace geom = lsst::afw::geom;

typedef float ImagePixelT;

BOOST_AUTO_TEST_CASE(FootprintSets) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    lsst::pex::logging::Trace::setVerbosity("afw.detection", 0);

    image::MaskedImage<ImagePixelT> img(geom::Extent2I(10,20));
    *img.getImage() = 100;

    detection::FootprintSet ds_by_value1(img, 0);
    BOOST_CHECK(ds_by_value1.getFootprints()->size() == 1);

    detection::FootprintSet ds_by_value2(img,
                                         detection::Threshold(0, detection::Threshold::VALUE));
    BOOST_CHECK(ds_by_value2.getFootprints()->size() == 1);

    BOOST_CHECK_THROW(detection::FootprintSet(img,         \
                                              detection::Threshold(0, detection::Threshold::STDEV)), \
                      lsst::pex::exceptions::Exception);
    
    BOOST_CHECK_THROW(detection::FootprintSet(img, \
                                              detection::Threshold(0, detection::Threshold::VARIANCE)), \
                      lsst::pex::exceptions::Exception);
}

/************************************************************************************************************/
/*
 * Find the sum of the pixels in a Footprint
 */
template <typename ImageT>
class PixelSum : public detection::FootprintFunctor<ImageT> {
public:
    PixelSum(ImageT const& mimage) :
        detection::FootprintFunctor<ImageT>(mimage),
        _counts(0.0)
        {}

    // method called for each pixel by apply()
    void operator()(typename ImageT::xy_locator loc, // locator pointing at the pixel
                    int, int
                   ) {
        _counts += *loc;
    }

    void reset() {
        _counts = 0.0;
    }
    virtual void reset(detection::Footprint const&) {}

    double getCounts() const { return _counts; }
private:
    typename ImageT::Pixel _counts;     // the sum of all pixels in the Footprint
};

BOOST_AUTO_TEST_CASE(FootprintFunctor) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    image::MaskedImage<ImagePixelT> mimg(geom::Extent2I(10,20));
    image::Image<ImagePixelT> img = *mimg.getImage();

    img = 0;
    img(5, 5) = 100;
    img(5, 10) = 100;

    detection::FootprintSet ds(mimg, 10);

    BOOST_CHECK(ds.getFootprints()->size() == 2);

    PixelSum<image::Image<ImagePixelT> > countDN(img);
    for (detection::FootprintSet::FootprintList::iterator ptr = ds.getFootprints()->begin(),
             end = ds.getFootprints()->end(); ptr != end; ++ptr) {
        countDN.apply(**ptr);
        BOOST_CHECK_CLOSE(countDN.getCounts(), 100.0, 1e-10);
    }
}
