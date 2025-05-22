/*
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ImageHashCpp
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop

#include <limits>

#include "lsst/cpputils/tests.h"

#include "lsst/afw/image.h"

namespace lsst {
namespace afw {
namespace image {

BOOST_AUTO_TEST_CASE(PixelHash) {
    using IntPixel = pixel::Pixel<int, int, double>;
    using FloatPixel = pixel::Pixel<double, int, double>;
    using IntSinglePixel = pixel::SinglePixel<int, int, double>;

    cpputils::assertValidHash<IntPixel>();
    cpputils::assertValidHash<FloatPixel>();
    cpputils::assertValidHash<IntSinglePixel>();

    cpputils::assertHashesEqual(IntPixel(42, 0, 1.0), IntPixel(42, 0, 1.0));
    cpputils::assertHashesEqual(FloatPixel(42.0, 0, 1.0), FloatPixel(42.0, 0, 1.0));
    // cpputils::assertHashesEqual(IntSinglePixel(42, 0, 1.0), IntSinglePixel(42, 0, 1.0));

    // Asymmetric cross-class equality needs some special handling
    BOOST_TEST_REQUIRE(IntPixel(42, 0, 1.0) == FloatPixel(42.0, 0, 1.0));
    BOOST_TEST(std::hash<IntPixel>()(IntPixel(42, 0, 1.0)) ==
               std::hash<FloatPixel>()(FloatPixel(42.0, 0, 1.0)));
    BOOST_TEST_REQUIRE(IntPixel(42, 0, 1.0) == IntSinglePixel(42, 0, 1.0));
    BOOST_TEST(std::hash<IntPixel>()(IntPixel(42, 0, 1.0)) ==
               std::hash<IntSinglePixel>()(IntSinglePixel(42, 0, 1.0)));
}

BOOST_AUTO_TEST_CASE(VisitInfoHash) {
    using lsst::daf::base::DateTime;
    using lsst::geom::degrees;

    cpputils::assertValidHash<VisitInfo>();

    // A builder would be really nice...
    VisitInfo info1(10.01, 11.02, DateTime(65321.1, DateTime::MJD, DateTime::TAI), 12345.1, 45.1 * degrees,
                    lsst::geom::SpherePoint(23.1 * degrees, 73.2 * degrees),
                    lsst::geom::SpherePoint(134.5 * degrees, 33.3 * degrees), 1.73, 73.2 * degrees,
                    RotType::SKY, coord::Observatory(11.1 * degrees, 22.2 * degrees, 0.333),
                    coord::Weather(1.1, 2.2, 34.5), "testCam1", 123456, 1.5, "test", "program", "reason",
                    "object", true);
    VisitInfo info2(10.01, 11.02, DateTime(65321.1, DateTime::MJD, DateTime::TAI), 12345.1, 45.1 * degrees,
                    lsst::geom::SpherePoint(23.1 * degrees, 73.2 * degrees),
                    lsst::geom::SpherePoint(134.5 * degrees, 33.3 * degrees), 1.73, 73.2 * degrees,
                    RotType::SKY, coord::Observatory(11.1 * degrees, 22.2 * degrees, 0.333),
                    coord::Weather(1.1, 2.2, 34.5), "testCam1", 123456, 1.5, "test", "program", "reason",
                    "object", true);

    cpputils::assertHashesEqual(info1, info2);
}

}  // namespace image
}  // namespace afw
}  // namespace lsst
