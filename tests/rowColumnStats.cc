// -*- lsst-c++ -*-

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

/*
 * An test executible which calls the statisticsStack function
 */
#include <iostream>
#include <memory>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE rowColumnStatistics

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/tools/floating_point_comparison.hpp"

#include "lsst/geom.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Stack.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

using ImageF = image::Image<float>;
using MImageF = image::MaskedImage<float>;

BOOST_AUTO_TEST_CASE(RowColumnStats) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25
                                          "Boost non-Std" */

    int const n = 8;

    // fill an image with a gradient
    std::vector<float> column(n, 0.0);
    std::vector<float> row(n, 0.0);
    std::shared_ptr<ImageF> img = std::make_shared<ImageF>(lsst::geom::Extent2I(n, n), 0);
    for (int y = 0; y < img->getHeight(); ++y) {
        int x = 0;
        for (ImageF::x_iterator ptr = img->row_begin(y), end = img->row_end(y); ptr != end; ++ptr, ++x) {
            *ptr = 1.0 * x + 2.0 * y;
            column[y] += *ptr;
            row[x] += *ptr;
            if (y == n - 1) {
                row[x] /= n;
            }
        }
        column[y] /= n;
    }

    // collapse with a MEAN over 'x' (ie. avg all columns to one), then 'y' (avg all rows to one)
    std::shared_ptr<MImageF> imgProjectCol = math::statisticsStack(*img, math::MEAN, 'x');
    std::shared_ptr<MImageF> imgProjectRow = math::statisticsStack(*img, math::MEAN, 'y');

    MImageF::x_iterator rPtr = imgProjectRow->row_begin(0);
    MImageF::y_iterator cPtr = imgProjectCol->col_begin(0);
    for (int i = 0; i < n; ++i, ++rPtr, ++cPtr) {
        BOOST_CHECK_EQUAL(cPtr.image(), column[i]);
        BOOST_CHECK_EQUAL(rPtr.image(), row[i]);
    }
}
