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

#ifndef LSST_AFW_TYPEHANDLING_TEST_H
#define LSST_AFW_TYPEHANDLING_TEST_H

#define BOOST_TEST_DYN_LINK
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop

#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>

#include <boost/mpl/list.hpp>

#include "lsst/pex/exceptions.h"

#include "lsst/afw/typehandling/GenericMap.h"
#include "lsst/afw/typehandling/Storable.h"

namespace lsst {
namespace afw {
namespace typehandling {
namespace test {

/*
 * This include file defines tests that exercise the GenericMap and MutableGenericMap interfaces, and ensures
 * that any implementation satisfies the requirements of these interfaces. Subclass authors should call either
 * addGenericMapTestCases or addMutableGenericMapTestCases in a suitable entry point, such as a global fixture
 * or a module initialization function.
 */

namespace {
class SimpleStorable : public Storable {
public:
    virtual ~SimpleStorable() = default;

    std::shared_ptr<Storable> cloneStorable() const override { return std::make_unique<SimpleStorable>(); }

    std::string toString() const override { return "Simplest possible representation"; }

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

    std::shared_ptr<Storable> cloneStorable() const override {
        return std::make_unique<ComplexStorable>(storage);
    }

    std::string toString() const override { return "ComplexStorable(" + std::to_string(storage) + ")"; }

    std::size_t hash_value() const noexcept override { return std::hash<double>()(storage); }

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

template <typename T>
std::string universalToString(T const& value) {
    std::stringstream buffer;
    buffer << value;
    return buffer.str();
};

// Would make more sense as static constants in GenericFactory
// but neither string nor Storable qualify as literal types
// In anonymous namespace to ensure constants are internal to whatever test includes this header
auto const KEY0 = makeKey<bool>(0);
bool const VALUE0 = true;
auto const KEY1 = makeKey<int>(1);
int const VALUE1 = 42;
auto const KEY2 = makeKey<double>(2);
int const VALUE2 = VALUE1;
auto const KEY3 = makeKey<std::string>(3);
std::string const VALUE3 = "How many roads must a man walk down?";
auto const KEY4 = makeKey<std::shared_ptr<SimpleStorable const>>(4);
auto const VALUE4 = SimpleStorable();
auto const KEY5 = makeKey<ComplexStorable>(5);
auto const VALUE5 = ComplexStorable(-100.0);
auto const KEY6 = makeKey<std::shared_ptr<Storable const>>(6);
auto const VALUE6 = std::shared_ptr<Storable const>();
}  // namespace

/**
 * Abstract factory that creates GenericMap and MutableGenericMap instances as needed.
 */
class GenericFactory {
public:
    virtual ~GenericFactory() = default;

    /**
     * Create a map containing the following state:
     *
     * * `KEY0: VALUE0`
     * * `KEY1: VALUE1`
     * * `KEY2: VALUE2`
     * * `KEY3: VALUE3`
     * * `KEY4: std::shared_ptr<>(VALUE4)`
     * * `KEY5: VALUE5`
     * * `KEY6: VALUE6`
     */
    virtual std::unique_ptr<GenericMap<int>> makeGenericMap() const = 0;

    /// Create an empty map.
    virtual std::unique_ptr<MutableGenericMap<std::string>> makeMutableGenericMap() const = 0;
};

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestConstAt, GenericMapFactory) {
    static GenericMapFactory const factory;
    std::unique_ptr<GenericMap<int> const> demoMap = factory.makeGenericMap();

    BOOST_TEST(demoMap->at(KEY0) == VALUE0);
    BOOST_TEST(demoMap->at(KEY1) == VALUE1);
    BOOST_TEST(demoMap->at(KEY2) == VALUE2);
    BOOST_TEST(demoMap->at(KEY3) == VALUE3);
    BOOST_TEST(*(demoMap->at(KEY4)) == VALUE4);
    BOOST_TEST(demoMap->at(KEY5) == VALUE5);
    BOOST_TEST(demoMap->at(KEY6) == VALUE6);
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestAt, GenericMapFactory) {
    using namespace std::string_literals;

    static GenericMapFactory const factory;
    std::unique_ptr<GenericMap<int>> demoMap = factory.makeGenericMap();

    BOOST_TEST(demoMap->at(KEY0) == VALUE0);
    demoMap->at(KEY0) = false;
    BOOST_TEST(demoMap->at(KEY0) == false);
    BOOST_CHECK_THROW(demoMap->at(makeKey<int>(KEY0.getId())), pex::exceptions::OutOfRangeError);

    BOOST_TEST(demoMap->at(KEY1) == VALUE1);
    demoMap->at(KEY1)++;
    BOOST_TEST(demoMap->at(KEY1) == VALUE1 + 1);
    BOOST_CHECK_THROW(demoMap->at(makeKey<bool>(KEY1.getId())), pex::exceptions::OutOfRangeError);

    BOOST_TEST(demoMap->at(KEY2) == VALUE2);
    demoMap->at(KEY2) = 0.0;
    BOOST_TEST(demoMap->at(KEY2) == 0.0);
    // VALUE2 is of a different type than KEY2, check that alternate key is absent
    using Type2 = std::remove_const_t<decltype(VALUE2)>;
    BOOST_CHECK_THROW(demoMap->at(makeKey<Type2>(KEY2.getId())), pex::exceptions::OutOfRangeError);

    BOOST_TEST(demoMap->at(KEY3) == VALUE3);
    demoMap->at(KEY3).append(" Oops, wrong question."s);
    BOOST_TEST(demoMap->at(KEY3) == VALUE3 + " Oops, wrong question."s);

    BOOST_TEST(*(demoMap->at(KEY4)) == VALUE4);
    // VALUE4 is of a different type than KEY4, check that alternate key is absent
    using Type4 = std::remove_const_t<decltype(VALUE4)>;
    BOOST_CHECK_THROW(demoMap->at(makeKey<Type4>(KEY4.getId())), pex::exceptions::OutOfRangeError);

    BOOST_TEST(demoMap->at(KEY5) == VALUE5);
    BOOST_TEST(demoMap->at(makeKey<SimpleStorable>(KEY5.getId())) == VALUE5);

    ComplexStorable newValue(5.0);
    demoMap->at(KEY5) = newValue;
    BOOST_TEST(demoMap->at(KEY5) == newValue);

    BOOST_TEST(demoMap->at(KEY6) == VALUE6);
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestEquals, GenericMapFactory) {
    static GenericMapFactory const factory;
    auto map1 = factory.makeGenericMap();

    // Use BOOST_CHECK to avoid BOOST_TEST bug from GenericMap being unprintable
    BOOST_CHECK(*map1 == *map1);
    // Maps are unequal because shared_ptr members point to different objects
    BOOST_CHECK(*map1 != *(factory.makeGenericMap()));
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestConstVisitor, GenericMapFactory) {
    static GenericMapFactory const factory;
    std::unique_ptr<GenericMap<int> const> const map = factory.makeGenericMap();
    std::vector<int> mapKeys = map->keys();

    // Visitors return string because it's one of the few operations valid for all types of interest
    // This lets us test generic lambdas as visitors
    auto bruteForcePrinter = [&map](int key) {
        switch (key) {
            case 0:
                return universalToString(map->at(KEY0));
            case 1:
                return universalToString(map->at(KEY1));
            case 2:
                return universalToString(map->at(KEY2));
            case 3:
                return universalToString(map->at(KEY3));
            case 4:
                return universalToString(map->at(KEY4));
            case 5:
                return universalToString(map->at(KEY5));
            case 6:
                return universalToString(map->at(KEY6));
            default:
                throw std::invalid_argument("Bad key found");
        };
    };
    std::vector<std::string> expected;
    for (int key : mapKeys) {
        expected.push_back(bruteForcePrinter(key));
    }

    // Test local class that returns void
    class {
    public:
        std::vector<std::string> results;

        // Local classes can't have method templates
        void operator()(int, bool value) { results.push_back(universalToString(value)); }
        void operator()(int, std::int32_t const& value) { results.push_back(universalToString(value)); }
        void operator()(int, std::int64_t value) { results.push_back(universalToString(value)); }
        void operator()(int, float const& value) { results.push_back(universalToString(value)); }
        void operator()(int, double value) { results.push_back(universalToString(value)); }
        void operator()(int, std::string const& value) { results.push_back(universalToString(value)); }
        void operator()(int, Storable const& value) { results.push_back(universalToString(value)); }
        void operator()(int, std::shared_ptr<Storable const> value) {
            results.push_back(universalToString(value));
        }
    } printer;
    map->apply(printer);
    BOOST_REQUIRE(printer.results.size() == expected.size());
    for (std::size_t i = 0; i < printer.results.size(); ++i) {
        BOOST_TEST(printer.results[i] == expected[i],
                   printer.results[i] << " != " << expected[i] << ", key = " << mapKeys[i]);
    }

    // Test lambda that returns string
    std::vector<std::string> strings =
            map->apply([](int, auto const& value) { return universalToString(value); });
    BOOST_REQUIRE(strings.size() == expected.size());
    for (std::size_t i = 0; i < strings.size(); ++i) {
        BOOST_TEST(strings[i] == expected[i],
                   strings[i] << " != " << expected[i] << ", key = " << mapKeys[i]);
    }
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestModifyingVoidVisitor, GenericMapFactory) {
    static GenericMapFactory const factory;
    std::unique_ptr<GenericMap<int>> map = factory.makeGenericMap();
    std::vector<int> originalKeys = map->keys();

    // Test local class that returns void
    class {
    public:
        // Local classes can't have method templates
        void operator()(int, bool& value) { value = !value; }
        void operator()(int, std::int32_t& value) { value *= 2; }
        void operator()(int, std::int64_t& value) { value *= 2; }
        void operator()(int, float& value) { value *= 2; }
        void operator()(int, double& value) { value *= 2; }
        void operator()(int, std::string& value) { value += "Appendix"; }
        void operator()(int, Storable& value) {
            auto complexStorable = dynamic_cast<ComplexStorable*>(&value);
            if (complexStorable != nullptr) {
                *complexStorable = 42;
            }
        }
        void operator()(int, std::shared_ptr<Storable const>) {}
    } grower;
    map->apply(grower);
    std::vector<int> newKeys = map->keys();

    BOOST_TEST(newKeys == originalKeys);
    BOOST_TEST(map->at(KEY0) == !VALUE0);
    BOOST_TEST(map->at(KEY1) == 2 * VALUE1);
    BOOST_TEST(map->at(KEY2) == 2 * VALUE2);
    BOOST_TEST(map->at(KEY3) == VALUE3 + "Appendix");
    BOOST_TEST(*(map->at(KEY4)) == VALUE4);
    BOOST_TEST(map->at(KEY5) != VALUE5);
    BOOST_TEST(map->at(KEY5) == ComplexStorable(42));
    BOOST_TEST(map->at(KEY6) == nullptr);
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestModifyingReturningVisitor, GenericMapFactory) {
    static GenericMapFactory const factory;
    std::unique_ptr<GenericMap<int>> map = factory.makeGenericMap();
    std::vector<int> originalKeys = map->keys();

    // Test local class that returns int
    class {
    public:
        // Local classes can't have method templates
        int operator()(int key, bool& value) {
            value = !value;
            return key;
        }
        int operator()(int key, std::int32_t& value) {
            value *= 2;
            return key;
        }
        int operator()(int key, std::int64_t& value) {
            value *= 2;
            return key;
        }
        int operator()(int key, float& value) {
            value *= 2;
            return key;
        }
        int operator()(int key, double& value) {
            value *= 2;
            return key;
        }
        int operator()(int key, std::string& value) {
            value += "Appendix";
            return key;
        }
        int operator()(int key, Storable& value) {
            auto complexStorable = dynamic_cast<ComplexStorable*>(&value);
            if (complexStorable != nullptr) {
                *complexStorable = 42;
            }
            return key;
        }
        int operator()(int key, std::shared_ptr<Storable const>) { return key; }
    } grower;
    std::vector<int> editedKeys = map->apply(grower);
    BOOST_TEST(editedKeys == originalKeys);

    std::vector<int> newKeys = map->keys();
    BOOST_TEST(newKeys == originalKeys);
    BOOST_TEST(map->at(KEY0) == !VALUE0);
    BOOST_TEST(map->at(KEY1) == 2 * VALUE1);
    BOOST_TEST(map->at(KEY2) == 2 * VALUE2);
    BOOST_TEST(map->at(KEY3) == VALUE3 + "Appendix");
    BOOST_TEST(*(map->at(KEY4)) == VALUE4);
    BOOST_TEST(map->at(KEY5) != VALUE5);
    BOOST_TEST(map->at(KEY5) == ComplexStorable(42));
    BOOST_TEST(map->at(KEY6) == nullptr);
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestMutableEquals, GenericMapFactory) {
    using namespace std::string_literals;

    static GenericMapFactory const factory;
    auto map1 = factory.makeMutableGenericMap();
    auto map2 = factory.makeMutableGenericMap();

    // Use BOOST_CHECK to avoid BOOST_TEST bug from GenericMap being unprintable
    BOOST_CHECK(*map1 == *map2);

    auto primitiveKey = makeKey<int>("primitive"s);
    map1->insert(primitiveKey, 42);
    BOOST_CHECK(*map1 != *map2);
    map2->insert(primitiveKey, 42);
    BOOST_CHECK(*map1 == *map2);

    auto sharedKey = makeKey<std::shared_ptr<SimpleStorable const>>("shared"s);
    auto common = std::make_shared<SimpleStorable const>(VALUE4);
    map1->insert(sharedKey, common);
    BOOST_CHECK(*map1 != *map2);
    map2->insert(sharedKey, std::make_shared<SimpleStorable const>(VALUE4));
    BOOST_CHECK(*map1 != *map2);
    map2->erase(sharedKey);
    map2->insert(sharedKey, common);
    BOOST_CHECK(*map1 == *map2);

    auto storableKey = makeKey<ComplexStorable>("storable"s);
    map1->insert(storableKey, VALUE5);
    BOOST_CHECK(*map1 != *map2);
    map2->insert(storableKey, VALUE5);
    BOOST_CHECK(*map1 == *map2);

    auto nullKey = makeKey<std::shared_ptr<ComplexStorable const>>("null"s);
    map1->insert(nullKey, std::static_pointer_cast<ComplexStorable const>(VALUE6));
    BOOST_CHECK(*map1 != *map2);
    map2->insert(nullKey, std::static_pointer_cast<ComplexStorable const>(VALUE6));
    BOOST_CHECK(*map1 == *map2);
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestSize, GenericMapFactory) {
    static GenericMapFactory const factory;
    std::unique_ptr<GenericMap<int>> demoMap = factory.makeGenericMap();

    BOOST_TEST(demoMap->size() == 7);
    BOOST_TEST(!demoMap->empty());
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestMutableSize, GenericMapFactory) {
    using namespace std::string_literals;

    static GenericMapFactory const factory;
    std::unique_ptr<MutableGenericMap<std::string>> demoMap = factory.makeMutableGenericMap();

    BOOST_TEST_REQUIRE(demoMap->size() == 0);
    BOOST_TEST_REQUIRE(demoMap->empty());

    demoMap->insert(makeKey<int>("Negative One"s), -1);
    BOOST_TEST(demoMap->size() == 1);
    BOOST_TEST(!demoMap->empty());

    demoMap->erase(makeKey<int>("Negative One"s));
    BOOST_TEST(demoMap->size() == 0);
    BOOST_TEST(demoMap->empty());
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestWeakContains, GenericMapFactory) {
    using namespace std::string_literals;

    static GenericMapFactory const factory;
    std::unique_ptr<GenericMap<int> const> demoMap = factory.makeGenericMap();

    BOOST_TEST(demoMap->contains(KEY0.getId()));
    BOOST_TEST(demoMap->contains(KEY1.getId()));
    BOOST_TEST(demoMap->contains(KEY2.getId()));
    BOOST_TEST(demoMap->contains(KEY3.getId()));
    BOOST_TEST(demoMap->contains(KEY4.getId()));
    BOOST_TEST(demoMap->contains(KEY5.getId()));
    BOOST_TEST(demoMap->contains(KEY6.getId()));
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestContains, GenericMapFactory) {
    static GenericMapFactory const factory;
    std::unique_ptr<GenericMap<int> const> demoMap = factory.makeGenericMap();

    BOOST_TEST(demoMap->contains(KEY0));
    BOOST_TEST(!demoMap->contains(makeKey<int>(KEY0.getId())));

    BOOST_TEST(demoMap->contains(KEY1));
    BOOST_TEST(!demoMap->contains(makeKey<bool>(KEY1.getId())));

    BOOST_TEST(demoMap->contains(KEY2));
    // VALUE2 is of a different type than KEY2, check that alternate key is absent
    BOOST_TEST(!demoMap->contains(makeKey<decltype(VALUE2)>(KEY2.getId())));

    BOOST_TEST(demoMap->contains(KEY3));

    BOOST_TEST(demoMap->contains(KEY4));
    // VALUE4 is of a different type than KEY4, check that alternate key is absent
    BOOST_TEST(!demoMap->contains(makeKey<decltype(VALUE4)>(KEY4.getId())));

    BOOST_TEST(demoMap->contains(KEY5));
    BOOST_TEST(demoMap->contains(makeKey<SimpleStorable>(KEY5.getId())));
    BOOST_TEST(demoMap->contains(makeKey<Storable>(KEY5.getId())));

    BOOST_TEST(demoMap->contains(KEY6));
    BOOST_TEST(demoMap->contains(makeKey<std::shared_ptr<SimpleStorable const>>(KEY6.getId())));
    BOOST_TEST(demoMap->contains(makeKey<std::shared_ptr<ComplexStorable const>>(KEY6.getId())));
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestKeys, GenericMapFactory) {
    static GenericMapFactory const factory;
    std::unique_ptr<GenericMap<int> const> demoMap = factory.makeGenericMap();
    auto orderedKeys = demoMap->keys();
    // GenericMap allows keys in any order, so just check they're the same
    std::set<int> keys(orderedKeys.begin(), orderedKeys.end());

    BOOST_TEST(keys == std::set<int>({KEY0.getId(), KEY1.getId(), KEY2.getId(), KEY3.getId(), KEY4.getId(),
                                      KEY5.getId(), KEY6.getId()}));
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestKeyOrder, GenericMapFactory) {
    static GenericMapFactory const factory;
    std::unique_ptr<GenericMap<int> const> demoMap = factory.makeGenericMap();
    auto keys = demoMap->keys();

    std::vector<int> iterOrder;
    demoMap->apply([&iterOrder](int key, auto value) { iterOrder.push_back(key); });
    BOOST_TEST(keys == iterOrder);
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestClearIdempotent, GenericMapFactory) {
    static GenericMapFactory const factory;
    std::unique_ptr<MutableGenericMap<std::string>> demoMap = factory.makeMutableGenericMap();

    BOOST_TEST_REQUIRE(demoMap->empty());
    demoMap->clear();
    BOOST_TEST(demoMap->empty());
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestClear, GenericMapFactory) {
    using namespace std::string_literals;

    static GenericMapFactory const factory;
    std::unique_ptr<MutableGenericMap<std::string>> demoMap = factory.makeMutableGenericMap();

    demoMap->insert(makeKey<int>("prime"s), 3);
    demoMap->insert(makeKey<std::string>("foo"s), "bar"s);

    BOOST_TEST_REQUIRE(!demoMap->empty());
    demoMap->clear();
    BOOST_TEST(demoMap->empty());
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestInsertInt, GenericMapFactory) {
    using namespace std::string_literals;

    static GenericMapFactory const factory;
    std::unique_ptr<MutableGenericMap<std::string>> demoMap = factory.makeMutableGenericMap();

    BOOST_TEST_REQUIRE(demoMap->empty());

    int x = 27;
    BOOST_TEST(demoMap->insert(makeKey<int>("cube"s), x) == true);
    BOOST_TEST(demoMap->insert(makeKey<int>("cube"s), 0) == false);

    BOOST_TEST(!demoMap->empty());
    BOOST_TEST(demoMap->size() == 1);
    BOOST_TEST(demoMap->contains("cube"s));
    BOOST_TEST(demoMap->contains(makeKey<int>("cube"s)));
    BOOST_TEST(!demoMap->contains(makeKey<double>("cube"s)));
    BOOST_TEST(demoMap->at(makeKey<int>("cube"s)) == x);

    x = 0;
    BOOST_TEST(demoMap->at(makeKey<int>("cube"s)) != x);

    demoMap->at(makeKey<int>("cube"s)) = 0;
    BOOST_TEST(demoMap->at(makeKey<int>("cube"s)) == 0);
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestWeakInsertInt, GenericMapFactory) {
    using namespace std::string_literals;

    static GenericMapFactory const factory;
    std::unique_ptr<MutableGenericMap<std::string>> demoMap = factory.makeMutableGenericMap();

    BOOST_TEST_REQUIRE(demoMap->empty());

    auto insertResult = demoMap->insert("cube"s, 27);
    BOOST_TEST(insertResult.second == true);
    BOOST_TEST(demoMap->insert("cube"s, 0).second == false);

    BOOST_TEST(!demoMap->empty());
    BOOST_TEST(demoMap->size() == 1);
    BOOST_TEST(demoMap->contains("cube"s));
    BOOST_TEST(demoMap->contains(insertResult.first));
    BOOST_TEST(demoMap->contains(makeKey<int>("cube"s)));
    BOOST_TEST(!demoMap->contains(makeKey<double>("cube"s)));
    BOOST_TEST(demoMap->at(insertResult.first) == 27);
    BOOST_TEST(demoMap->at(makeKey<int>("cube"s)) == 27);

    demoMap->at(insertResult.first) = 0;
    BOOST_TEST(demoMap->at(insertResult.first) == 0);
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestInsertString, GenericMapFactory) {
    using namespace std::string_literals;

    static GenericMapFactory const factory;
    std::unique_ptr<MutableGenericMap<std::string>> demoMap = factory.makeMutableGenericMap();

    BOOST_TEST_REQUIRE(demoMap->empty());

    std::string answer(
            "I have a most elegant and wonderful proof, but this string is too small to contain it."s);
    BOOST_TEST(demoMap->insert(makeKey<std::string>("Ultimate answer"s), answer) == true);
    BOOST_TEST(demoMap->insert(makeKey<std::string>("OK"s), "Ook!"s) == true);
    BOOST_TEST(demoMap->insert(makeKey<std::string>("Ultimate answer"s), "Something philosophical"s) ==
               false);

    BOOST_TEST(!demoMap->empty());
    BOOST_TEST(demoMap->size() == 2);
    BOOST_TEST(demoMap->contains("OK"s));
    BOOST_TEST(demoMap->contains(makeKey<std::string>("Ultimate answer"s)));
    BOOST_TEST(demoMap->at(makeKey<std::string>("Ultimate answer"s)) == answer);
    BOOST_TEST(demoMap->at(makeKey<std::string>("OK"s)) == "Ook!"s);

    answer = "I don't know"s;
    BOOST_TEST(demoMap->at(makeKey<std::string>("Ultimate answer"s)) != answer);
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestWeakInsertString, GenericMapFactory) {
    using namespace std::string_literals;

    static GenericMapFactory const factory;
    std::unique_ptr<MutableGenericMap<std::string>> demoMap = factory.makeMutableGenericMap();

    BOOST_TEST_REQUIRE(demoMap->empty());

    auto insertResult1 = demoMap->insert("Ultimate answer"s, "Something philosophical"s);
    BOOST_TEST(insertResult1.second == true);
    auto insertResult2 = demoMap->insert("OK"s, "Ook!"s);
    BOOST_TEST(insertResult2.second == true);

    BOOST_TEST(!demoMap->empty());
    BOOST_TEST(demoMap->size() == 2);
    BOOST_TEST(demoMap->contains(insertResult1.first));
    BOOST_TEST(demoMap->contains(insertResult2.first));
    BOOST_TEST(demoMap->contains("OK"s));
    BOOST_TEST(demoMap->contains(makeKey<std::string>("Ultimate answer"s)));
    BOOST_TEST(demoMap->at(insertResult1.first) == "Something philosophical"s);
    BOOST_TEST(demoMap->at(makeKey<std::string>("Ultimate answer"s)) == "Something philosophical"s);
    BOOST_TEST(demoMap->at(insertResult2.first) == "Ook!"s);
    BOOST_TEST(demoMap->at(makeKey<std::string>("OK"s)) == "Ook!"s);
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestInsertStorable, GenericMapFactory) {
    using namespace std::string_literals;

    static GenericMapFactory const factory;
    std::unique_ptr<MutableGenericMap<std::string>> demoMap = factory.makeMutableGenericMap();

    BOOST_TEST_REQUIRE(demoMap->empty());

    ComplexStorable object(3.1416);
    BOOST_TEST(demoMap->insert<Storable>(makeKey<Storable>("foo"s), object) == true);
    BOOST_TEST(demoMap->insert(makeKey<std::shared_ptr<ComplexStorable const>>("bar"s),
                               std::make_shared<ComplexStorable const>(3.141)) == true);
    BOOST_TEST(demoMap->insert<Storable>(makeKey<Storable>("foo"s), SimpleStorable()) == false);
    BOOST_TEST(demoMap->insert(makeKey<std::shared_ptr<SimpleStorable const>>("bar"s),
                               std::make_shared<SimpleStorable const>()) == false);
    BOOST_TEST(demoMap->insert(makeKey<std::shared_ptr<SimpleStorable const>>("null"s),
                               std::make_shared<SimpleStorable const>()) == true);

    BOOST_TEST(!demoMap->empty());
    BOOST_TEST(demoMap->size() == 3);
    BOOST_TEST(demoMap->contains("foo"s));
    BOOST_TEST(demoMap->contains(makeKey<Storable>("foo"s)));
    BOOST_TEST(demoMap->contains(makeKey<std::shared_ptr<ComplexStorable const>>("bar"s)));
    BOOST_TEST(demoMap->contains(makeKey<std::shared_ptr<SimpleStorable const>>("null"s)));

    // ComplexStorable::operator== is asymmetric
    BOOST_TEST(object == demoMap->at(makeKey<SimpleStorable>("foo"s)));
    object = ComplexStorable(1.4);
    BOOST_TEST(object != demoMap->at(makeKey<SimpleStorable>("foo"s)));
    BOOST_TEST(*(demoMap->at(makeKey<std::shared_ptr<ComplexStorable const>>("bar"s))) ==
               ComplexStorable(3.141));
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestInterleavedInserts, GenericMapFactory) {
    using namespace std::string_literals;

    static GenericMapFactory const factory;
    std::unique_ptr<MutableGenericMap<std::string>> demoMap = factory.makeMutableGenericMap();

    BOOST_TEST_REQUIRE(demoMap->empty());

    BOOST_TEST(demoMap->insert(makeKey<int>("key1"s), 3) == true);
    BOOST_TEST(demoMap->insert(makeKey<double>("key1"s), 1.0) == false);
    BOOST_TEST(demoMap->insert<Storable>(makeKey<Storable>("key2"s), SimpleStorable()) == true);
    BOOST_TEST(demoMap->insert(makeKey<std::string>("key3"s), "Test value"s) == true);
    BOOST_TEST(demoMap->insert(makeKey<std::string>("key4"s), "This is some text"s) == true);
    std::string const message = "Unknown value for key5."s;
    BOOST_TEST(demoMap->insert(makeKey<std::string>("key5"s), message) == true);
    BOOST_TEST(demoMap->insert(makeKey<int>("key3"s), 20) == false);
    BOOST_TEST(demoMap->insert<double>(makeKey<double>("key6"s), 42) == true);

    BOOST_TEST(!demoMap->empty());
    BOOST_TEST(demoMap->size() == 6);
    BOOST_TEST(demoMap->at(makeKey<int>("key1"s)) == 3);
    BOOST_TEST(demoMap->at(makeKey<double>("key6"s)) == 42);
    BOOST_TEST(demoMap->at(makeKey<SimpleStorable>("key2"s)) == SimpleStorable());
    BOOST_TEST(demoMap->at(makeKey<std::string>("key3"s)) == "Test value"s);
    BOOST_TEST(demoMap->at(makeKey<std::string>("key4"s)) == "This is some text"s);
    BOOST_TEST(demoMap->at(makeKey<std::string>("key5"s)) == message);
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestErase, GenericMapFactory) {
    using namespace std::string_literals;

    static GenericMapFactory const factory;
    std::unique_ptr<MutableGenericMap<std::string>> demoMap = factory.makeMutableGenericMap();

    demoMap->insert(makeKey<int>("Ultimate answer"s), 42);
    BOOST_TEST_REQUIRE(demoMap->size() == 1);

    BOOST_TEST(demoMap->erase(makeKey<std::string>("Ultimate answer"s)) == false);
    BOOST_TEST(demoMap->size() == 1);
    BOOST_TEST(demoMap->erase(makeKey<int>("Ultimate answer"s)) == true);
    BOOST_TEST(demoMap->size() == 0);
}

BOOST_TEST_CASE_TEMPLATE_FUNCTION(TestInsertEraseInsert, GenericMapFactory) {
    using namespace std::string_literals;

    static GenericMapFactory const factory;
    std::unique_ptr<MutableGenericMap<std::string>> demoMap = factory.makeMutableGenericMap();

    BOOST_TEST_REQUIRE(demoMap->empty());

    BOOST_TEST(demoMap->insert(makeKey<int>("Ultimate answer"s), 42) == true);
    BOOST_TEST(demoMap->insert(makeKey<int>("OK"s), 200) == true);
    BOOST_TEST(demoMap->erase(makeKey<int>("Ultimate answer"s)) == true);
    BOOST_TEST(demoMap->insert(makeKey<double>("Ultimate answer"s), 3.1415927) == true);

    BOOST_TEST(!demoMap->empty());
    BOOST_TEST(demoMap->size() == 2);
    BOOST_TEST(demoMap->contains("OK"s));
    BOOST_TEST(!demoMap->contains(makeKey<int>("Ultimate answer"s)));
    BOOST_TEST(demoMap->contains(makeKey<double>("Ultimate answer"s)));
    BOOST_TEST(demoMap->at(makeKey<double>("Ultimate answer"s)) == 3.1415927);
}

/**
 * Create generic test cases for a specific GenericMap implementation.
 *
 * @tparam GenericMapFactory a subclass of GenericFactory that creates the
 *      desired implementation. Must be default-constructible.
 * @param suite the test suite to add the tests to.
 */
template <class GenericMapFactory>
void addGenericMapTestCases(boost::unit_test::test_suite* const suite) {
    using factories = boost::mpl::list<GenericMapFactory>;

    suite->add(BOOST_TEST_CASE_TEMPLATE(TestConstAt, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestAt, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestEquals, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestSize, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestWeakContains, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestContains, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestKeys, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestKeyOrder, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestConstVisitor, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestModifyingVoidVisitor, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestModifyingReturningVisitor, factories));
}

/**
 * Create generic test cases for a specific MutableGenericMap implementation.
 *
 * The tests will include all those added by addGenericMapTestCases.
 *
 * @tparam GenericMapFactory a subclass of GenericFactory that creates the
 *      desired implementation. Must be default-constructible.
 * @param suite the test suite to add the tests to.
 */
template <class GenericMapFactory>
void addMutableGenericMapTestCases(boost::unit_test::test_suite* const suite) {
    using factories = boost::mpl::list<GenericMapFactory>;

    addGenericMapTestCases<GenericMapFactory>(suite);

    suite->add(BOOST_TEST_CASE_TEMPLATE(TestMutableEquals, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestMutableSize, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestClear, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestClearIdempotent, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestInsertInt, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestInsertString, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestInsertStorable, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestInterleavedInserts, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestErase, factories));
    suite->add(BOOST_TEST_CASE_TEMPLATE(TestInsertEraseInsert, factories));
}

/**
 * Create generic test cases for a specific GenericMap implementation.
 *
 * The tests will be added to the master test suite.
 *
 * @tparam GenericMapFactory a subclass of GenericFactory that creates the
 *      desired implementation. Must be default-constructible.
 */
template <class GenericMapFactory>
inline void addGenericMapTestCases() {
    addGenericMapTestCases<GenericMapFactory>(&(boost::unit_test::framework::master_test_suite()));
}

/**
 * Create generic test cases for a specific MutableGenericMap implementation.
 *
 * The tests will be added to the master test suite. They will include all
 * tests added by addGenericMapTestCases.
 *
 * @tparam GenericMapFactory a subclass of GenericFactory that creates the
 *      desired implementation. Must be default-constructible.
 */
template <class GenericMapFactory>
inline void addMutableGenericMapTestCases() {
    addMutableGenericMapTestCases<GenericMapFactory>(&(boost::unit_test::framework::master_test_suite()));
}

}  // namespace test
}  // namespace typehandling
}  // namespace afw
}  // namespace lsst

#endif
