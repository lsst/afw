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

#define BOOST_TEST_MODULE SchemaCpp

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/included/unit_test.hpp"
#pragma clang diagnostic pop

#include "lsst/utils/tests.h"

#include "lsst/afw/table/aggregates.h"
#include "lsst/afw/table/Schema.h"

/*
 * Unit tests for C++-only functionality in Schema.
 *
 * See test_schema.py for remaining unit tests.
 */
namespace lsst {
namespace afw {
namespace table {

BOOST_AUTO_TEST_CASE(Hash) {
    utils::assertValidHash<Schema>();

    // Schemas are equal even if they have different key names, documentation, units, and aliases
    Schema schema1;
    schema1.addField<int>("a_i", "", "pixels");
    schema1.addField<float>("a_f", "descriptive description");
    schema1.addField<lsst::geom::Angle>("a_a", "");
    schema1.getAliasMap()->set("a_AAA", "a_a");

    Schema schema2;
    schema2.addField<int>("b_i", "", "bovines");
    schema2.addField<float>("b_f", "non-descriptive description");
    schema2.addField<lsst::geom::Angle>("b_a", "");

    utils::assertHashesEqual(schema1, schema2);
}

}  // namespace table
}  // namespace afw
}  // namespace lsst
