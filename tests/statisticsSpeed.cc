// -*- LSST-C++ -*-

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

#include <iostream>
#include <limits>
#include <cmath>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE StatisticsSpeed

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"
#include "boost/timer.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Statistics.h"

using namespace std;

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;
namespace geom = lsst::afw::geom;

typedef image::Image<float> Image;


/*
 * This test verifies that turning off NanSafe will slow down the Statistics computation.
 *
 * It uses boost::timer to measure stats on an 8k x 8k image (ramp pixel values).
 * - The time for NanSafe = true should be slower
 */
BOOST_AUTO_TEST_CASE(StatisticsNanSafeSlower) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */

    // make a ramp image
    int const nx = 8192;
    int const ny = nx;
    Image imgSimple(geom::Extent2I(nx, ny));
    Image imgNanSafe(geom::Extent2I(nx, ny));
    Image imgMinMax(geom::Extent2I(nx, ny));
    double z0 = 10.0;
    double dzdx = 1.0;
    double mean = z0 + ((nx - 1.0)/2.0)*dzdx;
    double stdev = 0.0;
    for (int iY = 0; iY < ny; ++iY) {
        double x = 0;
        for (Image::x_iterator ptr = imgSimple.row_begin(iY); ptr != imgSimple.row_end(iY); ++ptr) {
            *ptr = z0 + dzdx*x;
            x += 1.0;
            stdev += (*ptr - mean)*(*ptr - mean);
        }
        x = 0;
        for (Image::x_iterator ptr = imgNanSafe.row_begin(iY); ptr != imgNanSafe.row_end(iY); ++ptr) {
            *ptr = 2.0*(z0 + dzdx*x);
            x += 1.0;
        }
        x = 0;
        for (Image::x_iterator ptr = imgMinMax.row_begin(iY); ptr != imgMinMax.row_end(iY); ++ptr) {
            *ptr = 3.0*(z0 + dzdx*x);
            x += 1.0;
        }
    }
    stdev = sqrt(stdev/(nx*ny - 1));

    boost::timer timer;

    {
        // turn off NanSafe - should be fastest
        math::StatisticsControl sctrl = math::StatisticsControl();
        sctrl.setNanSafe(false);
        timer.restart();
        math::Statistics statsSimple = math::makeStatistics(imgSimple, math::NPOINT | math::MEAN, sctrl);
        BOOST_CHECK_EQUAL(statsSimple.getValue(math::MEAN), mean);
        double tSimple = timer.elapsed();

        // turn on NanSafe
        sctrl.setNanSafe(true);
        timer.restart();
        math::Statistics statsNanSafe = math::makeStatistics(imgNanSafe, math::NPOINT | math::MEAN, sctrl);
        BOOST_CHECK_EQUAL(statsNanSafe.getValue(math::MEAN), 2*mean);
        double tNanSafe = timer.elapsed();

        // turn on max/min  - should be slowest
        sctrl.setNanSafe(true);
        timer.restart();
        math::Statistics statsMinMax =
            math::makeStatistics(imgMinMax, math::NPOINT | math::MEAN | math::MIN, sctrl);
        BOOST_CHECK_EQUAL(statsMinMax.getValue(math::MIN), 3*z0);
        double tMinMax = timer.elapsed();


        bool isFasterWithSimple = (tSimple < tNanSafe && tSimple < tMinMax);
        bool isSlowerWithMinMax = (tMinMax > tNanSafe && tMinMax > tSimple);

        std::cout << tSimple << " " << tNanSafe << " " << tMinMax << std::endl;

        if (! isFasterWithSimple) {
            std::cerr << "Warning: statistics were faster with nanSafe=true." << std::endl;
            std::cerr << "  This is should resolve with g++ >= 4.2, and opt=3" << std::endl;
        }
        if (! isSlowerWithMinMax) {
            std::cerr << "Warning: statistics were faster with min/max requested." << std::endl;
            std::cerr << "  This is should resolve with g++ >= 4.2, and opt=3" << std::endl;
        }

#if 0
        BOOST_CHECK(isFasterWithSimple);
        BOOST_CHECK(isSlowerWithMinMax);
#endif
    }

}


