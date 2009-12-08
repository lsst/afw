// -*- LSST-C++ -*-
#include <iostream>
#include <limits>
#include <cmath>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE StatisticsSpeed

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"
#include "boost/timer.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Statistics.h"

using namespace std;

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

typedef image::Image<float> Image;


/**
 * @brief This test verifies that turning off NanSafe will slow down the Statistics computation.
 *
 * It uses boost::timer to measure stats on an 8k x 8k image (ramp pixel values).
 * - The time for NanSafe = true should be slower
 */
BOOST_AUTO_TEST_CASE(StatisticsNanSafeSlower) {

    // make a ramp image 
    int const nx = 8192;
    int const ny = nx;
    Image img(nx, ny);
    double z0 = 10.0;
    double dzdx = 1.0;
    double mean = z0 + ((nx - 1.0)/2.0)*dzdx;
    double stdev = 0.0;
    for (int iY = 0; iY < ny; ++iY) {
        double x = 0;
        for (Image::x_iterator ptr = img.row_begin(iY); ptr != img.row_end(iY); ++ptr) {
            *ptr = z0 + dzdx*x;
            x += 1.0;
            stdev += (*ptr - mean)*(*ptr - mean);
        }
    }
    stdev = sqrt(stdev/(nx*ny - 1));

    boost::timer timer;

    {
        // turn off NanSafe - should be fastest
        math::StatisticsControl sctrl = math::StatisticsControl();
        sctrl.setNanSafe(false);
        timer.restart();
        math::Statistics statsSimple = math::makeStatistics(img, math::NPOINT | math::MEAN, sctrl);
        double tSimple = timer.elapsed();

        // turn on NanSafe
        img *= 2;
        sctrl.setNanSafe(true);
        timer.restart();
        math::Statistics statsNanSafe = math::makeStatistics(img, math::NPOINT | math::MEAN, sctrl);
        double tNanSafe = timer.elapsed();

        // turn on max/min  - should be slowest
        img *= 3;
        sctrl.setNanSafe(true);
        timer.restart();
        math::Statistics statsMinMax =
            math::makeStatistics(img, math::NPOINT | math::MEAN | math::MIN, sctrl);
        double tMinMax = timer.elapsed();

        bool isFasterWithSimple = (tSimple < tNanSafe && tSimple < tMinMax);
        bool isSlowerWithMinMax = (tMinMax > tNanSafe && tMinMax > tSimple);
        
        BOOST_CHECK_EQUAL(statsSimple.getValue(math::MEAN), mean);
        BOOST_CHECK_EQUAL(statsNanSafe.getValue(math::MEAN), 2*mean);
        BOOST_CHECK_EQUAL(statsMinMax.getValue(math::MIN), 3*2*z0);
        BOOST_CHECK(isFasterWithSimple);
        BOOST_CHECK(isSlowerWithMinMax);
    }

}


