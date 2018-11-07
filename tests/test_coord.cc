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
#define BOOST_TEST_MODULE CoordCpp
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop

#include "lsst/utils/tests.h"

#include "lsst/geom/Angle.h"
#include "lsst/afw/coord/Observatory.h"
#include "lsst/afw/coord/Weather.h"

namespace lsst {
namespace afw {
namespace coord {

BOOST_AUTO_TEST_CASE(ObservatoryHash) {
    using geom::radians;

    utils::assertValidHash<Observatory>();

    utils::assertHashesEqual(Observatory(((geom::TWOPI + 1.2) * radians).wrap(), 0.4 * radians, 5143.0),
                             Observatory((geom::TWOPI + 1.2) * radians, 0.4 * radians, 5143.0));
    utils::assertHashesEqual(Observatory(-0.3 * radians, ((0.4 + geom::TWOPI) * radians).wrap(), 716.0),
                             Observatory(-0.3 * radians, (0.4 + geom::TWOPI) * radians, 716.0));
}

BOOST_AUTO_TEST_CASE(WeatherHash) {
    utils::assertValidHash<Weather>();

    utils::assertHashesEqual(Weather(-5.0, 1.01e5, 70.0), Weather(-5.0, 1.01e5, 70.0));
}

}  // namespace coord
}  // namespace afw
}  // namespace lsst
