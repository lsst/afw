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

#define BOOST_TEST_MODULE PolymorphicValueCpp
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/included/unit_test.hpp"
#pragma clang diagnostic pop

#include "lsst/utils/tests.h"
#include "lsst/afw/typehandling/PolymorphicValue.h"
#include "lsst/afw/typehandling/test.h"

namespace lsst {
namespace afw {
namespace typehandling {

using test::ComplexStorable;
using test::SimpleStorable;

/**
 * Attempt to create an empty PolymorphicValue.
 *
 * There is no operation on PolymorphicValue that guarantees an empty state,
 * so test code should check the returned object.
 */
PolymorphicValue makeEmpty() {
    SimpleStorable foo;
    PolymorphicValue empty(foo);
    PolymorphicValue nonEmpty(std::move(empty));
    return empty;
}

BOOST_AUTO_TEST_CASE(Empty) {
    SimpleStorable foo;
    PolymorphicValue fooHolder(foo);
    BOOST_TEST(!fooHolder.empty());

    ComplexStorable bar(1.0);
    PolymorphicValue barHolder(bar);
    BOOST_TEST(!barHolder.empty());

    BOOST_WARN(!makeEmpty().empty());
}

BOOST_AUTO_TEST_CASE(Equals) {
    SimpleStorable foo1, foo2;
    ComplexStorable bar1(1.0), bar2(1.5);

    PolymorphicValue fooHolder1(foo1), fooHolder2(foo2);
    PolymorphicValue barHolder1(bar1), barHolder2(bar2);

    // Warning: operator== for SimpleStorable/ComplexStorable is ill-behaved; don't assume symmetry
    BOOST_CHECK_EQUAL(fooHolder1 == fooHolder2, foo1 == foo2);
    BOOST_CHECK_EQUAL(fooHolder2 == fooHolder1, foo2 == foo1);
    BOOST_CHECK_EQUAL(barHolder1 == barHolder2, bar1 == bar2);
    BOOST_CHECK_EQUAL(barHolder2 == barHolder1, bar2 == bar1);

    BOOST_CHECK_EQUAL(barHolder1 == fooHolder2, bar1 == foo2);
    BOOST_CHECK_EQUAL(barHolder2 == fooHolder1, bar2 == foo1);
    BOOST_CHECK_EQUAL(fooHolder1 == barHolder2, foo1 == bar2);
    BOOST_CHECK_EQUAL(fooHolder2 == barHolder1, foo2 == bar1);

    PolymorphicValue empty1 = makeEmpty(), empty2 = makeEmpty();
    // makeEmpty() can't guarantee the result is actually empty
    if (empty1.empty() && empty2.empty()) {
        BOOST_TEST(empty1 == empty2);
        BOOST_TEST(empty2 == empty1);
        BOOST_TEST(empty1 != fooHolder1);
        BOOST_TEST(fooHolder1 != empty1);
    } else {
        BOOST_WARN(!empty1.empty());
        BOOST_WARN(!empty2.empty());
    }
}

BOOST_AUTO_TEST_CASE(Hash) {
    utils::assertValidHash<PolymorphicValue>();

    utils::assertHashesEqual(PolymorphicValue(ComplexStorable(1.0)), PolymorphicValue(ComplexStorable(1.0)));
    utils::assertHashesEqual(PolymorphicValue(ComplexStorable(3.8)), PolymorphicValue(ComplexStorable(3.8)));

    auto unhashable = PolymorphicValue(SimpleStorable());
    BOOST_CHECK_THROW(std::hash<PolymorphicValue>()(unhashable), UnsupportedOperationException);
}

BOOST_AUTO_TEST_CASE(Copy) {
    auto original = PolymorphicValue(ComplexStorable(3.5));
    auto copy = PolymorphicValue(original);
    BOOST_CHECK(copy == original);

    // Independent copy?
    static_cast<ComplexStorable&>(copy.get()) = 4.2;
    BOOST_CHECK(copy != original);
}

BOOST_AUTO_TEST_CASE(Move) {
    auto original = PolymorphicValue(ComplexStorable(3.5));
    auto backup = PolymorphicValue(original);
    auto copy = PolymorphicValue(std::move(original));

    BOOST_CHECK(copy == backup);

    // changes to copy shouldn't affect original
    auto postMove = PolymorphicValue(original);
    BOOST_REQUIRE(original == postMove);
    static_cast<ComplexStorable&>(copy.get()) = 4.2;
    BOOST_CHECK(original == postMove);
}

BOOST_AUTO_TEST_CASE(CopyAssign) {
    auto original = PolymorphicValue(ComplexStorable(3.5));
    auto copy = PolymorphicValue(SimpleStorable());
    copy = original;
    BOOST_CHECK(copy == original);

    // Independent copy?
    static_cast<ComplexStorable&>(copy.get()) = 4.2;
    BOOST_CHECK(copy != original);
}

BOOST_AUTO_TEST_CASE(MoveAssign) {
    auto original = PolymorphicValue(ComplexStorable(3.5));
    auto backup = PolymorphicValue(original);
    auto copy = PolymorphicValue(SimpleStorable());
    copy = std::move(original);

    BOOST_CHECK(copy == backup);

    // changes to copy shouldn't affect original
    auto postMove = PolymorphicValue(original);
    BOOST_REQUIRE(original == postMove);
    static_cast<ComplexStorable&>(copy.get()) = 4.2;
    BOOST_CHECK(original == postMove);
}

BOOST_AUTO_TEST_CASE(ImplicitConversion) {
    auto original = ComplexStorable(3.5);
    PolymorphicValue holder = original;

    // Does PolymorphicValue make a copy?
    BOOST_REQUIRE(original.equals(holder.get()));
    static_cast<ComplexStorable&>(original) = 0.0;
    BOOST_CHECK(!original.equals(holder.get()));

    // Does modifying contents change holder?
    Storable& contents = holder;
    BOOST_REQUIRE(holder == PolymorphicValue(ComplexStorable(3.5)));
    static_cast<ComplexStorable&>(contents) = 1.6;
    BOOST_CHECK(holder != PolymorphicValue(ComplexStorable(3.5)));
}

}  // namespace typehandling
}  // namespace afw
}  // namespace lsst
