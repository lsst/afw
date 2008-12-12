#include <iostream>
#include <cmath>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Statistics

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Statistics.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

typedef image::Image<float> ImageT;

BOOST_AUTO_TEST_CASE(Statistics) {
    ImageT img(10,40);
    ImageT::Pixel const pixval = 10000;
    img = pixval;

    {
        math::Statistics<ImageT> stats = math::make_Statistics(img, math::NPOINT | math::STDEV | math::MEAN);
        double const mean = stats.getValue(math::MEAN);
        double const dmean = stats.getError(math::MEAN);
        double const sd = stats.getValue(math::STDEV);
        
        BOOST_CHECK_EQUAL(stats.getValue(math::NPOINT), img.getWidth()*img.getHeight());
        BOOST_CHECK_EQUAL(mean, img(0,0));
        BOOST_CHECK(std::isnan(dmean)); // we didn't ask for the error, so it's a NaN
        BOOST_CHECK_EQUAL(sd, 0);
    }

    {
        math::Statistics<ImageT> stats = math::make_Statistics(img, math::STDEV | math::MEAN | math::ERRORS);
        std::pair<double, double> const mean = stats.getResult(math::MEAN);
        double const sd = stats.getValue(math::STDEV);
        
        BOOST_CHECK_EQUAL(mean.first,  img(0,0));
        BOOST_CHECK_EQUAL(mean.second, sd/sqrt(img.getWidth()*img.getHeight()));
    }

    {
        math::Statistics<ImageT> stats = math::make_Statistics(img, math::NPOINT);
        BOOST_CHECK_THROW(stats.getValue(math::MEAN), lsst::pex::exceptions::InvalidParameter);
    }

    // ===============================================================================
    // sjb code for percentiles and clipped stats
    {
        math::Statistics<ImageT> stats = math::make_Statistics(img, math::MEDIAN);
        BOOST_CHECK_EQUAL(pixval, stats.getValue(math::MEDIAN));
    }
    {
        math::Statistics<ImageT> stats = math::make_Statistics(img, math::IQRANGE);
        BOOST_CHECK_EQUAL(0.0, stats.getValue(math::IQRANGE));
    }
    {
        math::Statistics<ImageT> stats = math::make_Statistics(img, math::MEANCLIP);
        BOOST_CHECK_EQUAL(pixval, stats.getValue(math::MEANCLIP));
    }
    {
        math::Statistics<ImageT> stats = math::make_Statistics(img, math::VARIANCECLIP);
        BOOST_CHECK_EQUAL(0.0, stats.getValue(math::VARIANCECLIP));
    }

    
    
    {
        ImageT img2(img);
        //
        // Add 1 to every other row, so the variance is 1/4
        //
        BOOST_REQUIRE(img.getHeight()%2 == 0);
        for (int y = 1; y < img.getHeight(); y += 2) {
            for (ImageT::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth(); ptr != end; ++ptr) {
                *ptr += 1;
            }
        }

        math::Statistics<ImageT> stats = math::make_Statistics(img2,
                                                               math::NPOINT | math::STDEV | math::MEAN | math::ERRORS);
        std::pair<double, double> const mean = stats.getResult(math::MEAN);
        double const n = stats.getValue(math::NPOINT);
        double const sd = stats.getValue(math::STDEV);
        
        BOOST_CHECK_EQUAL(mean.first,  img(0,0) + 0.5);
        BOOST_CHECK_EQUAL(sd, 1/sqrt(4.0)*sqrt(n/(n - 1)));
        BOOST_CHECK_CLOSE(mean.second, sd/sqrt(img.getWidth()*img.getHeight()), 1e-10);
    }
}
