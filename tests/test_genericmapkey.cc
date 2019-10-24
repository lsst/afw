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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GenericMapKeyCpp
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop

#include <string>
#include <sstream>

#include "lsst/utils/tests.h"

#include "lsst/afw/math/ChebyshevBoundedField.h"
#include "lsst/afw/typehandling/Key.h"

using namespace std::string_literals;

namespace lsst {
namespace afw {
namespace typehandling {

class Base {};
class Derived : public Base {};

// TODO: use templated test cases to streamline and generalize some of these tests

BOOST_AUTO_TEST_CASE(KeyCopies) {
    auto key1a = makeKey<double>("fancyKey"s);
    auto key1b(key1a);
    BOOST_TEST(key1a == key1b);

    auto key2a = makeKey<std::string>(101);
    auto key2b = key2a;
    BOOST_TEST(key2a == key2b);
}

BOOST_AUTO_TEST_CASE(KeyMoves) {
    auto key1a = makeKey<double>("fancyKey"s);
    auto key1b(key1a);
    BOOST_TEST(key1a == key1b);

    auto key2a = makeKey<std::string>(101);
    auto key2b = std::move(key2a);
    BOOST_TEST(key2b == makeKey<std::string>(101));
}

BOOST_AUTO_TEST_CASE(KeyId) {
    auto key1 = makeKey<int>("specialKey"s);
    std::string id1 = key1.getId();
    BOOST_TEST(id1 == "specialKey"s);

    auto key2 = makeKey<int>(26);
    int id2 = key2.getId();
    BOOST_TEST(id2 == 26);
}

BOOST_AUTO_TEST_CASE(KeyFactory) {
    Key<std::string, int> foo1 = makeKey<int>("foo"s);
    Key<std::string, int> foo2 = Key<std::string, int>("foo"s);
    BOOST_TEST(foo1 == foo2);
}

BOOST_AUTO_TEST_CASE(KeyOutputString) {
    std::stringstream buffer;
    buffer << makeKey<int>("fancyKey"s);
    // Don't test value in brackets; it's compiler-dependent
    BOOST_TEST(buffer.str().find("fancyKey<"s) == 0);
    BOOST_TEST(buffer.str().rfind(">"s) == buffer.str().size() - 1);
}

BOOST_AUTO_TEST_CASE(KeyOutputInt) {
    std::stringstream buffer;
    buffer << makeKey<std::string>(42);
    // Don't test value in brackets; it's compiler-dependent
    BOOST_TEST(buffer.str().find("42<"s) == 0);
    BOOST_TEST(buffer.str().rfind(">"s) == buffer.str().size() - 1);
}

BOOST_AUTO_TEST_CASE(KeyEquality) {
    auto key1 = makeKey<double>("fancyKey"s);
    auto key2 = makeKey<double>("fancyKey"s);
    auto key3 = makeKey<int>("fancyKey"s);
    auto key4 = makeKey<int const>("fancyKey"s);

    BOOST_TEST(key1 == key1);
    BOOST_TEST(!(key1 != key1));
    BOOST_TEST(key2 == key2);
    BOOST_TEST(!(key2 != key2));
    BOOST_TEST(key3 == key3);
    BOOST_TEST(!(key3 != key3));
    BOOST_TEST(key4 == key4);
    BOOST_TEST(!(key4 != key4));

    BOOST_TEST(key1 == key2);
    BOOST_TEST(!(key1 != key2));
    BOOST_TEST(key2 != key3);
    BOOST_TEST(!(key2 == key3));
    BOOST_TEST(key3 != key1);
    BOOST_TEST(!(key3 == key1));

    BOOST_TEST(key3 != key4);
    BOOST_TEST(!(key3 == key4));
}

BOOST_AUTO_TEST_CASE(KeySorting) {
    auto key1 = makeKey<double>("a"s);
    auto key2 = makeKey<int>("b"s);
    auto key3 = makeKey<double>("c"s);

    BOOST_TEST(key1 < key2);
    BOOST_TEST(key2 < key3);
    BOOST_TEST(key1 < key3);

    BOOST_TEST(std::less<>()(key1, key2));
    BOOST_TEST(std::less<>()(key2, key3));
    BOOST_TEST(std::less<>()(key1, key3));
}

BOOST_AUTO_TEST_CASE(KeyHash) {
    utils::assertValidHash<Key<std::string, int>>();
    utils::assertValidHash<Key<int, lsst::afw::math::ChebyshevBoundedField>>();

    using TestKey = Key<std::string, int>;
    utils::assertHashesEqual(TestKey("foo"s), TestKey("foo"s));
    utils::assertHashesEqual(TestKey("bar"s), makeKey<int>("bar"s));
}

BOOST_AUTO_TEST_CASE(KeyConvertPrimitives) {
    auto doubleKey = makeKey<double>("a"s);
    // Key<std::string, int> intKey = doubleKey;  // Does not compile; double&->int&
    Key<std::string, double const> constKey = doubleKey;
    // Key<std::string, double> doubleKey2 = constKey;  // Does not compile; lose const

    BOOST_TEST(doubleKey != constKey);
    BOOST_TEST(doubleKey.getId() == constKey.getId());
}

BOOST_AUTO_TEST_CASE(KeyConvertObjects) {
    auto baseKey = makeKey<Base>(42);
    // Key<int, Derived> downcastKey = baseKey;  // Does not compile; base->derived
    Key<int, Base const> constBaseKey = baseKey;
    // Key<int, Base> baseKey2 = constBaseKey;  // Does not compile; lose const
    auto derivedKey = makeKey<Derived const>(44);
    Key<int, Base const> upcastKey = derivedKey;
    // Key<int, Base> upcastKey2 = derivedKey;  // Does not compile; lose const

    BOOST_TEST(baseKey != constBaseKey);
    BOOST_TEST(baseKey.getId() == constBaseKey.getId());
    BOOST_TEST(upcastKey != derivedKey);
    BOOST_TEST(upcastKey.getId() == derivedKey.getId());
}

BOOST_AUTO_TEST_CASE(KeyConvertPointers) {
    using std::shared_ptr;

    auto derivedKey = makeKey<shared_ptr<Derived>>(42);
    Key<int, shared_ptr<Base>> upcastKey = derivedKey;
    Key<int, shared_ptr<Derived const>> constDerivedKey = derivedKey;
    Key<int, shared_ptr<Base const> const> constBaseKey = derivedKey;
    // Key<int, shared_ptr<Derived const>> downcastKey = constBaseKey;  // Does not compile; base->derived
    // Key<int, shared_ptr<Base>> baseKey2 = constBaseKey;           // Does not compile; lose const
    Key<int, shared_ptr<Base const>> constBaseKey2 = constBaseKey;  // Ok; implies copy of pointer

    BOOST_TEST(derivedKey != upcastKey);
    BOOST_TEST(derivedKey.getId() == upcastKey.getId());
    BOOST_TEST(derivedKey != constDerivedKey);
    BOOST_TEST(derivedKey.getId() == constDerivedKey.getId());
    BOOST_TEST(derivedKey != constBaseKey);
    BOOST_TEST(derivedKey.getId() == constBaseKey.getId());
    BOOST_TEST(constBaseKey2 != constBaseKey);
    BOOST_TEST(constBaseKey2.getId() == constBaseKey.getId());
}

}  // namespace typehandling
}  // namespace afw
}  // namespace lsst
