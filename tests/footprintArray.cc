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
#define BOOST_TEST_MODULE FootprintArray

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/detection.h"
#include "lsst/afw/detection/FootprintArray.cc"

#include "ndarray/eigen.h"

#include "Eigen/Core"

namespace image = lsst::afw::image;
namespace detection = lsst::afw::detection;
namespace geom = lsst::afw::geom;
namespace nd = ndarray;

void doRoundTrip(
    detection::Footprint const & footprint,
    nd::Array<double const,1,1> const & v,
    geom::Box2I const & box
) {
    nd::Array<double,2,2> i1 = detection::expandArray(footprint, v, box);
    nd::Array<double,1,1> v1 = detection::flattenArray(footprint, i1, box.getMin());
    BOOST_CHECK( std::equal(v1.begin(), v1.end(), v.begin()) );
}

BOOST_AUTO_TEST_CASE(conversion) {

    detection::Footprint footprint;
    footprint.addSpan(4, 3, 9);
    footprint.addSpan(5, 2, 8);
    footprint.addSpan(7, 4, 5);
    footprint.addSpan(7, 7, 9);
    footprint.addSpan(8, 2, 7);
    footprint.addSpan(9, 0, 5);

    int oldArea = footprint.getArea();

    footprint.normalize();

    BOOST_CHECK_EQUAL( oldArea, footprint.getArea() );

    nd::Array<double,1,1> v = nd::allocate(footprint.getArea());
    v.asEigen().setRandom();

    doRoundTrip(footprint, v, geom::Box2I(geom::Point2I(0, 0), geom::Extent2I(10, 10)));

    doRoundTrip(footprint, v, geom::Box2I(geom::Point2I(0, 2), geom::Extent2I(15, 12)));

    BOOST_CHECK_THROW( 
        detection::expandArray(footprint, v, geom::Box2I(geom::Point2I(1, 0), geom::Extent2I(9, 10))),
        lsst::pex::exceptions::InvalidParameterError
    );

    BOOST_CHECK_THROW( 
        detection::expandArray(footprint, v, geom::Box2I(geom::Point2I(0, 5), geom::Extent2I(10, 5))),
        lsst::pex::exceptions::InvalidParameterError
    );

    BOOST_CHECK_THROW( 
        detection::expandArray(footprint, v, geom::Box2I(geom::Point2I(0, 0), geom::Extent2I(9, 10))),
        lsst::pex::exceptions::InvalidParameterError
    );

    BOOST_CHECK_THROW( 
        detection::expandArray(footprint, v, geom::Box2I(geom::Point2I(0, 0), geom::Extent2I(10, 9))),
        lsst::pex::exceptions::InvalidParameterError
    );

    footprint.shift(20, 30);

    doRoundTrip(footprint, v, geom::Box2I(geom::Point2I(10, 15), geom::Extent2I(50, 60)));

    doRoundTrip(footprint, v, geom::Box2I(geom::Point2I(20, 30), geom::Extent2I(10, 10)));

}
