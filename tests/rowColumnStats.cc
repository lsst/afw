// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
/**
 * @file rowColumnStats.cc
 * @author Steve Bickerton
 * @brief An test executible which calls the statisticsStack function
 *
 */
#include <iostream>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE rowColumnStatistics

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Stack.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;
namespace geom = lsst::afw::geom;

typedef image::Image<float> ImageF;
typedef image::MaskedImage<float> MImageF;
typedef std::vector<float> VecF;
typedef boost::shared_ptr<VecF> VecFPtr;

BOOST_AUTO_TEST_CASE(RowColumnStats) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */

    int const n = 8;

    // fill an image with a gradient
    std::vector<float> column(n, 0.0);
    std::vector<float> row(n, 0.0);
    ImageF::Ptr img = ImageF::Ptr (new ImageF(geom::Extent2I(n, n), 0));
    for (int y = 0; y < img->getHeight(); ++y) {
        int x = 0;
        for (ImageF::x_iterator ptr = img->row_begin(y), end = img->row_end(y); ptr != end; ++ptr, ++x) {
            *ptr = 1.0*x + 2.0*y;
            column[y] += *ptr;
            row[x] += *ptr;
            if (y == n - 1) {
                row[x] /= n;
            }
        }
        column[y] /= n;
    }

    // collapse with a MEAN over 'x' (ie. avg all columns to one), then 'y' (avg all rows to one)
    MImageF::Ptr imgProjectCol = math::statisticsStack(*img, math::MEAN, 'x');
    MImageF::Ptr imgProjectRow = math::statisticsStack(*img, math::MEAN, 'y');
    
    MImageF::x_iterator rPtr = imgProjectRow->row_begin(0);
    MImageF::y_iterator cPtr = imgProjectCol->col_begin(0);
    for (int i = 0; i < n; ++i, ++rPtr, ++cPtr) {
        BOOST_CHECK_EQUAL(cPtr.image(), column[i]);
        BOOST_CHECK_EQUAL(rPtr.image(), row[i]);
    }
}
