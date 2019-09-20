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
#define BOOST_TEST_MODULE StorableMapCpp
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop

#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>

#include "lsst/pex/exceptions.h"
#include "lsst/utils/tests.h"

#include "lsst/afw/typehandling/Key.h"
#include "lsst/afw/image/detail/StorableMap.h"

using namespace std;
using namespace string_literals;

namespace lsst {
namespace afw {
namespace image {
namespace detail {

namespace {
class SimpleStorable : public typehandling::Storable {
public:
    virtual ~SimpleStorable() = default;

    shared_ptr<typehandling::Storable> cloneStorable() const override {
        return make_shared<SimpleStorable>();
    }

    string toString() const override { return "Simplest possible representation"; }

    bool equals(Storable const& other) const noexcept override { return singleClassEquals(*this, other); }
    virtual bool operator==(SimpleStorable const& other) const { return true; }
    bool operator!=(SimpleStorable const& other) const { return !(*this == other); }
};

class ComplexStorable final : public SimpleStorable {
public:
    constexpr ComplexStorable(double storage) : SimpleStorable(), storage(storage) {}

    ComplexStorable& operator=(double newValue) {
        storage = newValue;
        return *this;
    }

    shared_ptr<typehandling::Storable> cloneStorable() const override {
        return make_shared<ComplexStorable>(storage);
    }

    string toString() const override { return "ComplexStorable(" + to_string(storage) + ")"; }

    size_t hash_value() const noexcept override { return hash<double>()(storage); }

    // Warning: violates both substitution and equality symmetry!
    bool equals(Storable const& other) const noexcept override {
        auto complexOther = dynamic_cast<ComplexStorable const*>(&other);
        if (complexOther) {
            return this->storage == complexOther->storage;
        } else {
            return false;
        }
    }
    bool operator==(SimpleStorable const& other) const override { return this->equals(other); }

private:
    double storage;
};

class OtherStorable final : public typehandling::Storable {};

auto const KEY_SIMPLE = typehandling::makeKey<shared_ptr<SimpleStorable const>>("key0"s);
auto const VALUE_SIMPLE = make_shared<SimpleStorable const>();
auto const KEY_COMPLEX = typehandling::makeKey<shared_ptr<ComplexStorable const>>("key1"s);
auto const VALUE_COMPLEX = make_shared<ComplexStorable const>(-100.0);
auto const KEY_NULL = typehandling::makeKey<shared_ptr<typehandling::Storable const>>("key2"s);
auto const VALUE_NULL = shared_ptr<typehandling::Storable const>();
auto const KEY_MIXED = typehandling::makeKey<shared_ptr<SimpleStorable const>>("key3"s);
auto const VALUE_MIXED = make_shared<ComplexStorable const>(42.0);
auto const KEY_BAD = typehandling::makeKey<shared_ptr<typehandling::Storable const>>("NotAKey"s);

unique_ptr<StorableMap const> makePrefilledMap() {
    // Can't use auto here
    initializer_list<StorableMap::value_type> contents = {
            make_pair(KEY_SIMPLE, VALUE_SIMPLE), make_pair(KEY_COMPLEX, VALUE_COMPLEX),
            make_pair(KEY_NULL, VALUE_NULL), make_pair(KEY_MIXED, VALUE_MIXED)};
    return make_unique<StorableMap>(contents);
}

}  // namespace

BOOST_AUTO_TEST_CASE(TestEquals) {
    auto map1 = makePrefilledMap();

    // Use BOOST_CHECK to avoid BOOST_TEST bug from StorableMap being unprintable
    BOOST_CHECK(*map1 == *map1);
}

BOOST_AUTO_TEST_CASE(TestSize) {
    auto demoMap = makePrefilledMap();

    BOOST_TEST(demoMap->size() == 4);
    BOOST_TEST(!demoMap->empty());
}

BOOST_AUTO_TEST_CASE(TestWeakContains) {
    auto demoMap = makePrefilledMap();

    BOOST_TEST(demoMap->contains(KEY_SIMPLE.getId()));
    BOOST_TEST(demoMap->contains(KEY_COMPLEX.getId()));
    BOOST_TEST(demoMap->contains(KEY_NULL.getId()));
    BOOST_TEST(demoMap->contains(KEY_MIXED.getId()));
    BOOST_TEST(!demoMap->contains(KEY_BAD.getId()));
}

BOOST_AUTO_TEST_CASE(TestClearIdempotent) {
    auto demoMap = make_unique<StorableMap>();

    BOOST_TEST_REQUIRE(demoMap->empty());
    demoMap->clear();
    BOOST_TEST(demoMap->empty());
}

}  // namespace detail
}  // namespace image
}  // namespace afw
}  // namespace lsst
