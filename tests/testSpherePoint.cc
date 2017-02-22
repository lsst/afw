// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See COPYRIGHT file at the top of the source tree.
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program. If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SpherePointCpp

#include "boost/test/unit_test.hpp"

#include "lsst/afw/geom/SpherePoint.h"
#include "lsst/pex/exceptions/Exception.h"

/*
 * Unit tests for C++-only functionality in SpherePoint.
 *
 * See testSpherePoint.py for remaining unit tests.
 */
namespace lsst {
namespace afw {
namespace geom {

/**
 * Tests whether the result of SpherePoint::SpherePoint(SpherePoint const &)
 * makes an identical but independent copy.
 */
BOOST_AUTO_TEST_CASE(SpherePointCopyResult, *boost::unit_test::tolerance(1e-14)) {
    SpherePoint original(Point3D(0.34, -1.2, 0.97));
    SpherePoint copy(original);

    // Want exact equality, not floating-point equality
    BOOST_TEST(original == copy);

    // Don't compare Angles in case there's aliasing of some sort
    double const copyLon = copy.getLongitude().asDegrees();
    double const copyLat = copy.getLatitude().asDegrees();

    original = SpherePoint(-42 * degrees, 45 * degrees);
    BOOST_TEST(original != copy);
    BOOST_TEST(copy.getLongitude().asDegrees() == copyLon);
    BOOST_TEST(copy.getLatitude().asDegrees() == copyLat);
}

/**
 * Tests whether SpherePoint::operator[](size_t) handles invalid indices
 * correctly.
 */
BOOST_AUTO_TEST_CASE(getItemError) {
    SpherePoint point(Point3D(1.0, 1.0, 1.0));

    BOOST_CHECK_THROW(point[2], pex::exceptions::OutOfRangeError);
    BOOST_CHECK_THROW(point[-1], pex::exceptions::OutOfRangeError);
}

// TODO: add a test for propagation of ostream errors
}
}
} /* namespace lsst::afw::geom */
