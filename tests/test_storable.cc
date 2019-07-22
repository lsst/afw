// -*- LSST-C++ -*-
/*
 * This file is part of afw.
 *
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

#define BOOST_TEST_MODULE StorableCpp
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/included/unit_test.hpp"
#pragma clang diagnostic pop

#include <sstream>

#include "lsst/afw/typehandling/Storable.h"

namespace lsst {
namespace afw {
namespace typehandling {

namespace {

class Dummy : public Storable {};

class Comparable : public Storable {
public:
    explicit Comparable(int id) : id(id) {}

    bool operator==(Comparable const& other) const noexcept { return id == other.id; }
    bool operator!=(Comparable const& other) const noexcept { return !(*this == other); }

    bool equals(Storable const& other) const noexcept override { return singleClassEquals(*this, other); }

private:
    int id;
};

}  // namespace

BOOST_AUTO_TEST_CASE(Defaults) {
    Dummy dummy;
    BOOST_CHECK_THROW(dummy.cloneStorable(), UnsupportedOperationException);

    BOOST_TEST(!dummy.isPersistable());
    BOOST_TEST(dummy.equals(dummy));
    BOOST_TEST(!dummy.equals(Dummy()));

    std::stringstream buffer;
    BOOST_CHECK_THROW(buffer << dummy, UnsupportedOperationException);
}

BOOST_AUTO_TEST_CASE(SingleClassEquals) {
    Comparable value1(42), value2(44);

    BOOST_REQUIRE(value1 == value1);
    BOOST_REQUIRE(value2 == value2);
    BOOST_REQUIRE(value1 != value2);
    BOOST_REQUIRE(value2 != value1);

    BOOST_TEST(value1.equals(value1));
    BOOST_TEST(value2.equals(value2));
    BOOST_TEST(!value1.equals(value2));
    BOOST_TEST(!value2.equals(value1));

    Dummy dummy;
    BOOST_TEST(!value1.equals(dummy));
    BOOST_TEST(!value2.equals(dummy));
    BOOST_TEST(!dummy.equals(value1));
    BOOST_TEST(!dummy.equals(value1));
}

}  // namespace typehandling
}  // namespace afw
}  // namespace lsst
