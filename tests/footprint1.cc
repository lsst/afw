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

#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Footprint

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/log/Log.h"
#include "lsst/afw/detection.h"

namespace image = lsst::afw::image;
namespace detection = lsst::afw::detection;
namespace geom = lsst::afw::geom;

typedef float ImagePixelT;

BOOST_AUTO_TEST_CASE(FootprintSets) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    LOG_CONFIG();
    LOG_SET_LVL("afw.detection", LOG_LVL_INFO);

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
