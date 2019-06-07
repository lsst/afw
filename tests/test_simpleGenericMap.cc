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
#define BOOST_TEST_MODULE SimpleGenericMapCpp
#define BOOST_TEST_NO_MAIN
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop

#include <memory>
#include <string>

#include "lsst/afw/typehandling/SimpleGenericMap.h"
#include "lsst/afw/typehandling/test.h"

using namespace std::string_literals;

namespace lsst {
namespace afw {
namespace typehandling {

class SimpleGenericMapFactory final : public test::GenericFactory {
public:
    /**
     * Create a map containing the following state:
     *
     * * `KEY0: VALUE0`
     * * `KEY1: VALUE1`
     * * `KEY2: VALUE2`
     * * `KEY3: VALUE3`
     * * `KEY4: std::shared_ptr<>(VALUE4)`
     * * `KEY5: VALUE5`
     */
    virtual std::unique_ptr<GenericMap<int>> makeGenericMap() const {
        auto map = std::make_unique<SimpleGenericMap<int>>();
        map->insert(test::KEY0, test::VALUE0);
        map->insert(test::KEY1, test::VALUE1);
        map->insert<double>(test::KEY2, test::VALUE2);
        map->insert(test::KEY3, test::VALUE3);
        map->insert(test::KEY4, std::make_shared<test::SimpleStorable>(test::VALUE4));
        map->insert(test::KEY5, test::VALUE5);
        return map;
    }

    /// Create an empty map.
    virtual std::unique_ptr<MutableGenericMap<std::string>> makeMutableGenericMap() const {
        return std::make_unique<SimpleGenericMap<std::string>>();
    }
};

std::unique_ptr<SimpleGenericMap<int>> makeDerivedMap() {
    static SimpleGenericMapFactory const factory;
    using Map = SimpleGenericMap<int>;
    // Exception-safe because only makeGenericMap() can throw
    return std::unique_ptr<Map>(dynamic_cast<Map*>(factory.makeGenericMap().release()));
}

void checkIndependentCopy(MutableGenericMap<int>& copy, MutableGenericMap<int>& original) {
    // Use BOOST_CHECK to avoid BOOST_TEST bug from GenericMap being unprintable
    BOOST_CHECK(original == copy);

    // copy and original should change independently
    copy.erase(test::KEY1);
    BOOST_CHECK(copy != original);
    copy.insert(test::KEY1, test::VALUE1);
    BOOST_CHECK(copy == original);
    copy.at(test::KEY0) = !test::VALUE0;
    BOOST_CHECK(copy != original);
    original.at(test::KEY0) = !test::VALUE0;
    BOOST_CHECK(copy == original);
}

BOOST_AUTO_TEST_CASE(Copy) {
    auto original = makeDerivedMap();

    SimpleGenericMap<int> copy(*original);
    checkIndependentCopy(copy, *original);
}

BOOST_AUTO_TEST_CASE(CopyConvert) {
    std::unique_ptr<MutableGenericMap<int>> original = makeDerivedMap();

    SimpleGenericMap<int> copy(*original);
    checkIndependentCopy(copy, *original);
}

BOOST_AUTO_TEST_CASE(CopyAssign) {
    auto original = makeDerivedMap();

    SimpleGenericMap<int> copy;
    copy = *original;
    checkIndependentCopy(copy, *original);
}

BOOST_AUTO_TEST_CASE(CopyAssignConvert) {
    std::unique_ptr<MutableGenericMap<int>> original = makeDerivedMap();

    SimpleGenericMap<int> copy;
    copy = *original;
    checkIndependentCopy(copy, *original);
}

void checkIndependentMove(SimpleGenericMap<int>& copy, SimpleGenericMap<int> const& original) {
    // state of original is undefined, but changes to copy should not affect it
    SimpleGenericMap<int> const movedValue(original);
    BOOST_REQUIRE(movedValue == original);

    copy.insert(makeKey<bool>(-127), true);
    BOOST_CHECK(movedValue == original);
    copy.clear();
    BOOST_CHECK(movedValue == original);
    copy.insert(makeKey<bool>(-127), true);
    BOOST_CHECK(movedValue == original);
}

BOOST_AUTO_TEST_CASE(Move) {
    auto original = makeDerivedMap();
    SimpleGenericMap<int> const backup(*original);

    SimpleGenericMap<int> copy(std::move(*original));
    BOOST_CHECK(backup == copy);

    checkIndependentMove(copy, *original);
}

BOOST_AUTO_TEST_CASE(MoveAssign) {
    auto original = makeDerivedMap();
    SimpleGenericMap<int> const backup(*original);

    SimpleGenericMap<int> copy;
    copy = std::move(*original);
    BOOST_CHECK(backup == copy);

    checkIndependentMove(copy, *original);
}

BOOST_AUTO_TEST_CASE(IterationOrder) {
    using namespace std::string_literals;
    static SimpleGenericMapFactory const factory;
    auto map = factory.makeMutableGenericMap();
    auto const& keys = map->keys();

    BOOST_REQUIRE(map->insert(makeKey<int>("firstKey"s), 42) == true);
    BOOST_REQUIRE(map->insert(makeKey<std::string>("secondKey"s), "someValue"s) == true);
    BOOST_REQUIRE(map->insert(makeKey<test::ComplexStorable>("thirdKey"s), test::ComplexStorable(-2.0)) ==
                  true);
    // Failed insert should not change iteration order
    BOOST_REQUIRE(map->insert(makeKey<int>("firstKey"s), 0) == false);
    BOOST_REQUIRE(map->insert(makeKey<std::string>("fourthKey"s), "anotherValue"s) == true);
    // Failed insert should not change iteration order
    BOOST_REQUIRE(map->insert(makeKey<double>("thirdKey"s), -2.0) == false);
    BOOST_REQUIRE(map->insert(makeKey<bool>("fifthKey"s), false) == true);
    BOOST_REQUIRE(map->erase(makeKey<std::string>("secondKey"s)) == true);
    BOOST_REQUIRE(map->erase(makeKey<std::string>("fourthKey"s)) == true);
    // A re-inserted key should not remember old position
    BOOST_REQUIRE(map->insert(makeKey<std::string>("secondKey"s), "someValue"s) == true);

    BOOST_TEST(keys == std::vector<std::string>({"firstKey"s, "thirdKey"s, "fifthKey"s, "secondKey"s}));
}

/// Boost::test initialization function
// Yes, we have to do it this way if we want the GenericMap tests to be defined
// before we have a concrete GenericMap class
// https://www.boost.org/doc/libs/1_68_0/libs/test/doc/html/boost_test/tests_organization/test_cases/test_organization_templates.html#ref_BOOST_TEST_CASE_TEMPLATE
bool init_unit_test() {
    test::addMutableGenericMapTestCases<SimpleGenericMapFactory>();
    return true;
}

}  // namespace typehandling
}  // namespace afw
}  // namespace lsst

/// Boost::test entry point
// Must be customized to call our init_unit_test
// https://www.boost.org/doc/libs/1_68_0/libs/test/doc/html/boost_test/adv_scenarios/shared_lib_customizations/init_func.html
int main(int argc, char* argv[]) {
    return boost::unit_test::unit_test_main(&lsst::afw::typehandling::init_unit_test, argc, argv);
}
