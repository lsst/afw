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

#define BOOST_TEST_MODULE CameraSysCpp
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/included/unit_test.hpp"
#pragma clang diagnostic pop

#include "lsst/utils/tests.h"

#include "lsst/afw/cameraGeom/CameraSys.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

BOOST_AUTO_TEST_CASE(Hash) {
    utils::assertValidHash<CameraSysPrefix>();

    utils::assertHashesEqual(PIXELS, CameraSysPrefix("Pixels"));
    utils::assertHashesEqual(ACTUAL_PIXELS, CameraSysPrefix("ActualPixels"));

    utils::assertValidHash<CameraSys>();

    utils::assertHashesEqual(FOCAL_PLANE, CameraSys("FocalPlane"));
    utils::assertHashesEqual(FIELD_ANGLE, CameraSys("FieldAngle"));
}

}  // namespace cameraGeom
}  // namespace afw
}  // namespace lsst
