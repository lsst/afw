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

#define BOOST_TEST_MODULE KeyCpp

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/included/unit_test.hpp"
#pragma clang diagnostic pop

#include "lsst/utils/tests.h"

#include "lsst/geom/Angle.h"
#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/Flag.h"
#include "lsst/afw/table/Schema.h"

/*
 * Unit tests for C++-only functionality in Key.
 */
namespace lsst {
namespace afw {
namespace table {

BOOST_AUTO_TEST_CASE(Hash) {
    utils::assertValidHash<Key<lsst::geom::Angle>>();
    utils::assertValidHash<Key<std::string>>();
    utils::assertValidHash<Key<Flag>>();

    Schema schema;
    auto key1 = schema.addField<lsst::geom::Angle>("key1", "");
    auto key2 = schema.addField<std::string>("key2", "", "", 42);
    auto key3 = schema.addField<Flag>("key3", "");

    utils::assertHashesEqual(key1, schema.find<lsst::geom::Angle>("key1").key);
    utils::assertHashesEqual(key2, schema.find<std::string>("key2").key);
    utils::assertHashesEqual(key3, schema.find<Flag>("key3").key);
}

}  // namespace table
}  // namespace afw
}  // namespace lsst
