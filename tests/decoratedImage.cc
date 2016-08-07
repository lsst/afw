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
#include <string>
#include <algorithm>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE DecoratedImage

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/Image.h"

namespace daf_base = lsst::daf::base;
namespace image = lsst::afw::image;
namespace geom = lsst::afw::geom;

using namespace std;

typedef float PixelT;
typedef image::Image<PixelT> ImageT;
typedef image::DecoratedImage<PixelT> DecoratedImageT;

/************************************************************************************************************/

DecoratedImageT make_image(int const width=5, int const height=6) {
    DecoratedImageT dimg(geom::Extent2I(width, height));
    ImageT::Ptr img = dimg.getImage();

    int i = 0;
    for (ImageT::iterator ptr = img->begin(), end = img->end(); ptr != end; ++ptr, ++i) {
        *ptr = i/dimg.getWidth() + 100*(i%dimg.getWidth());
    }

    return dimg;
}

/************************************************************************************************************/

BOOST_AUTO_TEST_CASE(setValues) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    DecoratedImageT dimg = make_image();
    daf_base::PropertySet::Ptr metadata = dimg.getMetadata();

    metadata->add("RHL", 1);
}
