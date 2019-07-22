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

#define BOOST_TEST_MODULE PolygonCpp
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/included/unit_test.hpp"
#pragma clang diagnostic pop

#include "lsst/utils/tests.h"

#include "lsst/geom/Point.h"
#include "lsst/afw/geom/polygon/Polygon.h"

namespace lsst {
namespace afw {
namespace geom {
namespace polygon {

BOOST_AUTO_TEST_CASE(Hash) {
    using lsst::geom::Point2D;

    utils::assertValidHash<Polygon>();

    utils::assertHashesEqual(Polygon({Point2D(-1.0, -1.0), Point2D(-1.0, 1.0), Point2D(1.0, 1.0)}),
                             Polygon({Point2D(-1.0, -1.0), Point2D(-1.0, 1.0), Point2D(1.0, 1.0)}));
    utils::assertHashesEqual(
            Polygon({Point2D(-1.0, -1.0), Point2D(-1.0, 1.0), Point2D(1.0, 1.0), Point2D(1.0, -1.0)}),
            Polygon(lsst::geom::Box2D(Point2D(-1.0, -1.0), lsst::geom::Extent2D(2.0, 2.0))));
}

}  // namespace polygon
}  // namespace geom
}  // namespace afw
}  // namespace lsst
