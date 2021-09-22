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
    ~SimpleStorable() override = default;

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

BOOST_AUTO_TEST_CASE(TestAt) {
    auto demoMap = makePrefilledMap();

    BOOST_TEST(demoMap->at(KEY_SIMPLE) == VALUE_SIMPLE);

    BOOST_TEST(demoMap->at(KEY_COMPLEX) == VALUE_COMPLEX);
    BOOST_TEST(demoMap->at(typehandling::makeKey<shared_ptr<typehandling::Storable const>>(
                       KEY_COMPLEX.getId())) == VALUE_COMPLEX);

    BOOST_TEST(demoMap->at(KEY_NULL) == VALUE_NULL);

    BOOST_TEST(demoMap->at(KEY_MIXED) == VALUE_MIXED);
    using ExactType = std::decay_t<decltype(VALUE_MIXED)>;
    BOOST_TEST(demoMap->at(typehandling::makeKey<ExactType>(KEY_MIXED.getId())) == VALUE_MIXED);

    BOOST_CHECK_THROW(
            demoMap->at(typehandling::makeKey<shared_ptr<ComplexStorable const>>(KEY_SIMPLE.getId())),
            pex::exceptions::OutOfRangeError);
    BOOST_CHECK_THROW(demoMap->at(KEY_BAD), pex::exceptions::OutOfRangeError);

    // None of these should compile, because they're not shared pointers to Storable const.
    // demoMap->at(typehandling::makeKey<int>("InvalidKey"s));
    // demoMap->at(typehandling::makeKey<SimpleStorable>("InvalidKey"s));
    // demoMap->at(typehandling::makeKey<shared_ptr<SimpleStorable>>("InvalidKey"s));
    // demoMap->at(typehandling::makeKey<shared_ptr<string const>>("InvalidKey"s));
}

BOOST_AUTO_TEST_CASE(TestEquals) {
    auto map1 = makePrefilledMap();

    // Use BOOST_CHECK to avoid BOOST_TEST bug from StorableMap being unprintable
    BOOST_CHECK(*map1 == *map1);
    // Test unequal maps in TestMutableEquals
}

BOOST_AUTO_TEST_CASE(TestMutableEquals) {
    auto map1 = make_unique<StorableMap>();
    auto map2 = make_unique<StorableMap>();

    // Use BOOST_CHECK to avoid BOOST_TEST bug from StorableMap being unprintable
    BOOST_CHECK(*map1 == *map2);

    auto sharedKey = typehandling::makeKey<shared_ptr<SimpleStorable const>>("simple"s);
    auto common = shared_ptr<SimpleStorable const>(VALUE_SIMPLE);
    map1->insert(sharedKey, common);
    BOOST_CHECK(*map1 != *map2);
    map2->insert(sharedKey, make_shared<SimpleStorable const>(*VALUE_SIMPLE));
    BOOST_CHECK(*map1 != *map2);
    map2->erase(sharedKey);
    map2->insert(sharedKey, common);
    BOOST_CHECK(*map1 == *map2);

    auto storableKey = typehandling::makeKey<shared_ptr<ComplexStorable const>>("complex"s);
    map1->insert<ComplexStorable const>(storableKey, VALUE_COMPLEX);
    BOOST_CHECK(*map1 != *map2);
    map2->insert<typehandling::Storable const>(
            typehandling::makeKey<shared_ptr<typehandling::Storable const>>(storableKey.getId()),
            VALUE_COMPLEX);
    BOOST_CHECK(*map1 == *map2);

    auto nullKey = typehandling::makeKey<shared_ptr<ComplexStorable const>>("null"s);
    map1->insert(nullKey, static_pointer_cast<ComplexStorable const>(VALUE_NULL));
    BOOST_CHECK(*map1 != *map2);
    map2->insert(nullKey, static_pointer_cast<ComplexStorable const>(VALUE_NULL));
    BOOST_CHECK(*map1 == *map2);
}

BOOST_AUTO_TEST_CASE(TestSize) {
    auto demoMap = makePrefilledMap();

    BOOST_TEST(demoMap->size() == 4);
    BOOST_TEST(!demoMap->empty());
}

BOOST_AUTO_TEST_CASE(TestMutableSize) {
    auto demoMap = make_unique<StorableMap>();

    BOOST_TEST_REQUIRE(demoMap->size() == 0);
    BOOST_TEST_REQUIRE(demoMap->empty());

    demoMap->insert(typehandling::makeKey<shared_ptr<SimpleStorable const>>("Simply Storable"s),
                    make_shared<SimpleStorable const>());
    BOOST_TEST(demoMap->size() == 1);
    BOOST_TEST(!demoMap->empty());

    demoMap->erase(typehandling::makeKey<shared_ptr<SimpleStorable const>>("Simply Storable"s));
    BOOST_TEST(demoMap->size() == 0);
    BOOST_TEST(demoMap->empty());
}

BOOST_AUTO_TEST_CASE(TestWeakContains) {
    auto demoMap = makePrefilledMap();

    BOOST_TEST(demoMap->contains(KEY_SIMPLE.getId()));
    BOOST_TEST(demoMap->contains(KEY_COMPLEX.getId()));
    BOOST_TEST(demoMap->contains(KEY_NULL.getId()));
    BOOST_TEST(demoMap->contains(KEY_MIXED.getId()));
    BOOST_TEST(!demoMap->contains(KEY_BAD.getId()));
}

BOOST_AUTO_TEST_CASE(TestContains) {
    auto demoMap = makePrefilledMap();

    BOOST_TEST(demoMap->contains(KEY_SIMPLE));
    BOOST_TEST(
            !demoMap->contains(typehandling::makeKey<shared_ptr<OtherStorable const>>(KEY_SIMPLE.getId())));
    BOOST_TEST(
            !demoMap->contains(typehandling::makeKey<shared_ptr<ComplexStorable const>>(KEY_SIMPLE.getId())));

    BOOST_TEST(demoMap->contains(KEY_COMPLEX));
    BOOST_TEST(demoMap->contains(
            typehandling::makeKey<shared_ptr<typehandling::Storable const>>(KEY_COMPLEX.getId())));

    BOOST_TEST(demoMap->contains(KEY_NULL));
    // While KEY_NULL did not insert ComplexStorable, the value is nullptr, which is valid shared_ptr<Complex>
    BOOST_TEST(demoMap->contains(typehandling::makeKey<shared_ptr<ComplexStorable const>>(KEY_NULL.getId())));

    BOOST_TEST(demoMap->contains(KEY_MIXED));
    using ExactType = std::decay_t<decltype(VALUE_MIXED)>;
    BOOST_TEST(demoMap->contains(typehandling::makeKey<ExactType>(KEY_MIXED.getId())));

    BOOST_TEST(!demoMap->contains(KEY_BAD));

    // None of these should compile, because they're not shared pointers to Storable const.
    // demoMap->contains(typehandling::makeKey<int>("InvalidKey"s));
    // demoMap->contains(typehandling::makeKey<SimpleStorable>("InvalidKey"s));
    // demoMap->contains(typehandling::makeKey<shared_ptr<SimpleStorable>>("InvalidKey"s));
    // demoMap->contains(typehandling::makeKey<shared_ptr<string const>>("InvalidKey"s));
}

BOOST_AUTO_TEST_CASE(TestClearIdempotent) {
    auto demoMap = make_unique<StorableMap>();

    BOOST_TEST_REQUIRE(demoMap->empty());
    demoMap->clear();
    BOOST_TEST(demoMap->empty());
}

BOOST_AUTO_TEST_CASE(TestClear) {
    auto demoMap = make_unique<StorableMap>();

    demoMap->insert(typehandling::makeKey<shared_ptr<ComplexStorable const>>("prime"s),
                    make_shared<ComplexStorable const>(3.0));
    demoMap->insert(typehandling::makeKey<shared_ptr<SimpleStorable const>>("foo"s),
                    make_shared<SimpleStorable const>());

    BOOST_TEST_REQUIRE(!demoMap->empty());
    demoMap->clear();
    BOOST_TEST(demoMap->empty());
}

BOOST_AUTO_TEST_CASE(TestInsertStorable) {
    auto demoMap = make_unique<StorableMap>();

    BOOST_TEST_REQUIRE(demoMap->empty());

    auto pointer = make_shared<ComplexStorable>(3.1416);
    BOOST_TEST(demoMap->insert<typehandling::Storable const>(
                       typehandling::makeKey<shared_ptr<typehandling::Storable const>>("foo"s), pointer) ==
               true);
    BOOST_TEST(demoMap->insert(typehandling::makeKey<shared_ptr<ComplexStorable const>>("bar"s),
                               make_shared<ComplexStorable const>(2.72)) == true);
    BOOST_TEST(demoMap->insert<typehandling::Storable const>(
                       typehandling::makeKey<shared_ptr<typehandling::Storable const>>("foo"s),
                       make_shared<SimpleStorable>()) == false);
    BOOST_TEST(demoMap->insert(typehandling::makeKey<shared_ptr<SimpleStorable const>>("bar"s),
                               make_shared<SimpleStorable const>()) == false);
    BOOST_TEST(demoMap->insert(typehandling::makeKey<shared_ptr<SimpleStorable const>>("null"s),
                               make_shared<SimpleStorable const>()) == true);

    BOOST_TEST(!demoMap->empty());
    BOOST_TEST(demoMap->size() == 3);
    BOOST_TEST(demoMap->contains("foo"s));
    BOOST_TEST(demoMap->contains(typehandling::makeKey<shared_ptr<typehandling::Storable const>>("foo"s)));
    BOOST_TEST(demoMap->contains(typehandling::makeKey<shared_ptr<ComplexStorable const>>("bar"s)));
    BOOST_TEST(demoMap->contains(typehandling::makeKey<shared_ptr<SimpleStorable const>>("null"s)));

    // ComplexStorable::operator== is asymmetric
    BOOST_TEST(*pointer == *(demoMap->at(typehandling::makeKey<shared_ptr<SimpleStorable const>>("foo"s))));
    pointer.reset(new ComplexStorable(1.4));
    BOOST_TEST(*pointer != *(demoMap->at(typehandling::makeKey<shared_ptr<SimpleStorable const>>("foo"s))));
    BOOST_TEST(ComplexStorable(3.1416) ==
               *(demoMap->at(typehandling::makeKey<shared_ptr<SimpleStorable const>>("foo"s))));
    BOOST_TEST(*(demoMap->at(typehandling::makeKey<shared_ptr<ComplexStorable const>>("bar"s))) ==
               ComplexStorable(2.72));

    // None of these should compile, because they're not shared pointers to Storable const.
    // demoMap->insert(typehandling::makeKey<int>("InvalidKey"s), 42);
    // demoMap->insert(typehandling::makeKey<SimpleStorable>("InvalidKey"s), SimpleStorable());
    // demoMap->insert(typehandling::makeKey<shared_ptr<SimpleStorable>>("InvalidKey"s),
    //                 make_shared<SimpleStorable>());
    // demoMap->insert(typehandling::makeKey<shared_ptr<string const>>("InvalidKey"s),
    //                 make_shared<string const>("Pointy string"));
    // demoMap->insert("InvalidKey"s, 42);
    // demoMap->insert("InvalidKey"s, SimpleStorable());
    // demoMap->insert("InvalidKey"s, make_shared<SimpleStorable>());
    // demoMap->insert("InvalidKey"s, make_shared<string const>("Pointy string"));
}

BOOST_AUTO_TEST_CASE(TestInterleavedInserts) {
    auto demoMap = make_unique<StorableMap>();

    BOOST_TEST_REQUIRE(demoMap->empty());

    BOOST_TEST(demoMap->insert("key1"s, make_shared<SimpleStorable const>()).second == true);
    BOOST_TEST(demoMap->insert("key1"s, make_shared<ComplexStorable const>(1.0)).second == false);
    BOOST_TEST(demoMap->insert<typehandling::Storable const>(
                       typehandling::makeKey<shared_ptr<typehandling::Storable const>>("key2"s),
                       make_shared<SimpleStorable>()) == true);
    BOOST_TEST(demoMap->insert("key3"s, make_shared<ComplexStorable const>(3.0)).second == true);
    BOOST_TEST(demoMap->insert("key4"s, make_shared<ComplexStorable const>(42.0)).second == true);
    BOOST_TEST(demoMap->insert("key3"s, make_shared<OtherStorable const>()).second == false);

    BOOST_TEST(!demoMap->empty());
    BOOST_TEST(demoMap->size() == 4);
    BOOST_TEST(*(demoMap->at(typehandling::makeKey<shared_ptr<ComplexStorable const>>("key4"s))) ==
               ComplexStorable(42.0));
    BOOST_TEST(*(demoMap->at(typehandling::makeKey<shared_ptr<SimpleStorable const>>("key1"s))) ==
               SimpleStorable());
    BOOST_TEST(*(demoMap->at(typehandling::makeKey<shared_ptr<ComplexStorable const>>("key3"s))) ==
               ComplexStorable(3.0));
    BOOST_TEST(*(demoMap->at(typehandling::makeKey<shared_ptr<SimpleStorable const>>("key2"s))) ==
               SimpleStorable());
}

BOOST_AUTO_TEST_CASE(TestErase) {
    auto demoMap = make_unique<StorableMap>();

    demoMap->insert(typehandling::makeKey<shared_ptr<ComplexStorable const>>("Ultimate answer"s),
                    make_shared<ComplexStorable const>(42.0));
    BOOST_TEST_REQUIRE(demoMap->size() == 1);

    BOOST_TEST(demoMap->erase(typehandling::makeKey<shared_ptr<OtherStorable const>>("Ultimate answer"s)) ==
               false);
    BOOST_TEST(demoMap->size() == 1);
    BOOST_TEST(demoMap->erase(typehandling::makeKey<shared_ptr<ComplexStorable const>>("Ultimate answer"s)) ==
               true);
    BOOST_TEST(demoMap->size() == 0);
}

BOOST_AUTO_TEST_CASE(TestInsertEraseInsert) {
    static double const PI = 3.1415927;

    auto demoMap = make_unique<StorableMap>();

    BOOST_TEST_REQUIRE(demoMap->empty());

    BOOST_TEST(demoMap->insert("Changing type"s, make_shared<OtherStorable const>()).second == true);
    BOOST_TEST(demoMap->insert("Extra"s, make_shared<SimpleStorable const>()).second == true);
    BOOST_TEST(demoMap->erase(typehandling::makeKey<shared_ptr<OtherStorable const>>("Changing type"s)) ==
               true);
    BOOST_TEST(demoMap->insert("Changing type"s, make_shared<ComplexStorable const>(PI)).second == true);

    BOOST_TEST(!demoMap->empty());
    BOOST_TEST(demoMap->size() == 2);
    BOOST_TEST(demoMap->contains("Extra"s));
    BOOST_TEST(!demoMap->contains(typehandling::makeKey<shared_ptr<OtherStorable const>>("Changing type"s)));
    BOOST_TEST(demoMap->contains(typehandling::makeKey<shared_ptr<ComplexStorable const>>("Changing type"s)));
    BOOST_TEST(*(demoMap->at(typehandling::makeKey<shared_ptr<ComplexStorable const>>("Changing type"s))) ==
               ComplexStorable(PI));
}

BOOST_AUTO_TEST_CASE(TestIteration) {
    // Copy to get a non-const map.
    StorableMap map = *makePrefilledMap();

    shared_ptr<typehandling::Storable const> dummy = make_shared<OtherStorable>();

    for (auto& keyValue : map) {
        auto& value = keyValue.second;

        if (value == nullptr) {
            value = dummy;
        }
    }

    for (auto it = begin(map); it != end(map); ++it) {
        auto& key = it->first;
        auto& value = it->second;

        if (key == KEY_NULL) {
            BOOST_TEST(value == dummy);
        } else {
            BOOST_TEST(value != dummy);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestConstIteration) {
    unique_ptr<StorableMap const> map = makePrefilledMap();

    unordered_map<string, string> expected({make_pair(KEY_SIMPLE.getId(), VALUE_SIMPLE->toString()),
                                            make_pair(KEY_COMPLEX.getId(), VALUE_COMPLEX->toString()),
                                            make_pair(KEY_NULL.getId(), "null"),
                                            make_pair(KEY_MIXED.getId(), VALUE_MIXED->toString())});

    unordered_map<string, string> result;
    for (auto it = cbegin(*map); it != cend(*map); ++it) {
        auto const& key = it->first;
        auto const& value = it->second;

        result.emplace(key.getId(), value ? value->toString() : "null");
    }

    BOOST_TEST(expected == result);
}

}  // namespace detail
}  // namespace image
}  // namespace afw
}  // namespace lsst
