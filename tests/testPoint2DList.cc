// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See COPYRIGHT file at the top of the source tree.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program. If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#include <functional>
#include <initializer_list>
#include <limits>
#include <list>
#include <set>
#include <vector>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Point2DListCpp

#include "boost/test/unit_test.hpp"

#include "lsst/afw/geom/Point2DList.h"
#include "lsst/pex/exceptions.h"

namespace pexExcept = lsst::pex::exceptions;
using namespace std;

/*
 * Unit tests for Point2DList.
 *
 * Many of these tests should be translated to Python once a Pybind11 interface is available.
 */
namespace lsst {
namespace afw {
namespace geom {

/**
 * Fixture for test cases that don't test initialization
 *
 * Boost dataset framework would be better, but for some reason references to
 *     BOOST_DATA_TEST_CASE won't compile.
 */
struct Data {
    Data() : lists() {
        Point2D filler(5.0, 3.0);

        lists.emplace_back();
        lists.emplace_back(std::initializer_list<Point2D>{filler});
        lists.emplace_back(
                std::initializer_list<Point2D>{Point2D(-3.0, 5.4), Point2D(8.0, 8.0), Point2D(4.5, 6.0)});
        lists.emplace_back(100, filler);
    }

    std::list<std::vector<Point2D>> lists;
};

/// Tests whether the default constructor creates an empty list.
BOOST_AUTO_TEST_CASE(Point2DListDefault) {
    Point2DList defaultList;
    BOOST_TEST(defaultList.empty());
}

/**
 * Returns the maximum size of a Point2DList.
 *
 * This method assumes the maximum size is the same for all `Point2DList`
 * objects in a given program build.
 *
 * @return The maximum allowed size for a `Point2DList`.
 */
Point2DList::size_type getMaxSize() { return Point2DList().max_size(); }

/**
 * Tests whether a constructor call rejects extremely long lists.
 *
 * The test takes into account whether it is possible to request a too-long
 * list on the platform on which the test is being run, and prints a
 * message if the test does not apply.
 *
 * @param construct A function that accepts the list length to request and
 *                  calls a Point2DList constructor. Must propagate all
 *                  exceptions to the caller.
 * @param signature A human-friendly signature for the constructor called
 *                  by `construct`.
 */
void checkMaxConstructor(function<void(Point2DList::size_type)> const &construct, string const &signature) {
    static auto sizeLimit = numeric_limits<Point2DList::size_type>::max();
    auto max = getMaxSize();

    if (max < sizeLimit) {
        BOOST_CHECK_THROW(construct(max + 1), pexExcept::LengthError);
    } else {
        BOOST_WARN_MESSAGE(false, "All possible arguments to " + signature + " are legal on this platform.");
    }
}

/**
 * Tests whether a method call prevents extremely long lists.
 *
 * @param method A function that accepts a list and modifies it in a way that
 *               would exceed the maximum size. Must propagate all exceptions
 *               to the caller. May assume the argument is a list of maximum
 *               size.
 */
void checkMaxMethod(function<void(Point2DList &)> const &method) {
    auto max = getMaxSize();
    Point2DList testbed(max);
    BOOST_CHECK_THROW(method(testbed), pexExcept::LengthError);
}

/**
 * Tests whether a constructor accepting `size_type` rejects
 *        impossibly long lists.
 */
BOOST_AUTO_TEST_CASE(Point2DListZeroedErrorCheck, *boost::unit_test::disabled()) {
    checkMaxConstructor([](Point2DList::size_type n) { Point2DList x(n); }, "Point2DList(size_type)");
}

/**
 * Tests whether a constructor accepting only `size_type` creates a
 *        zeroed list of the expected length.
 */
BOOST_AUTO_TEST_CASE(Point2DListZeroed, *boost::unit_test::disabled()) {
    Point2DList zeroedList(100);
    BOOST_TEST(zeroedList.size() == 100u);
    // TODO: use iterators once Point2DList::*iterator implemented
    // for (Point2D const point : zeroedList) {
    for (size_t i = 0; i < zeroedList.size(); ++i) {
        auto point = zeroedList[i];
        BOOST_TEST(point[0] == 0);
        BOOST_TEST(point[1] == 0);
    }
}

/**
 * Tests whether a constructor accepting `size_type` rejects
 *        impossibly long lists.
 */
BOOST_AUTO_TEST_CASE(Point2DListFilledErrorCheck, *boost::unit_test::disabled()) {
    Point2D filler(-2.0, 3.5);

    checkMaxConstructor([filler](Point2DList::size_type n) { Point2DList x(n, filler); },
                        "Point2DList(size_type, Point2D)");
}

/**
 * Tests whether a constructor that copies an element creates a
 *        copied list of the expected length.
 */
BOOST_AUTO_TEST_CASE(Point2DListFilled, *boost::unit_test::disabled()) {
    Point2D filler(-2.0, 3.5);

    Point2DList filledList(100, filler);
    BOOST_TEST(filledList.size() == 100u);
    // TODO: use iterators once Point2DList::*iterator implemented
    // for (Point2D const point : filledList) {
    for (size_t i = 0; i < filledList.size(); ++i) {
        auto point = filledList[i];
        BOOST_TEST(point == filler);
    }
}

/// Tests whether the container constructor rejects impossibly long lists.
BOOST_AUTO_TEST_CASE(Point2DListIteratorErrorCheck) {
    using BigInt = unsigned long long;
    static auto sizeLimit = numeric_limits<BigInt>::max();
    auto max = getMaxSize();

    if (max < sizeLimit) {
        BigInt size = static_cast<BigInt>(max) + 1;
        // std::array and other containers may have too-tight size bounds
        auto rawArray = unique_ptr<Point2D[]>(new Point2D[size]);
        BOOST_CHECK_THROW(Point2DList(rawArray.get(), rawArray.get() + size), pexExcept::LengthError);
    } else {
        BOOST_WARN_MESSAGE(
                false, "Cannot create arrays longer than the longest valid Point2DList on this platform.");
    }
}

/**
 * Tests whether two containers have the same elements in the same
 *        iteration order.
 *
 * @tparam T,U Containers supporting the standard `size` method, and
 *             either the standard `begin` and `end` methods or overloads
 *             to `std::begin` and `std::end`.
 *
 * @param t,u Containers of types `T` and `U`, respectively.
 */
template <class T, class U>
void checkElements(T const &t, U const &u) {
    BOOST_TEST_REQUIRE(t.size() == u.size());
    typename T::const_iterator itT;
    typename U::const_iterator itU;
    for (itT = begin(t), itU = begin(u); itT != end(t) && itU != end(u); ++itT, ++itU) {
        BOOST_TEST(*itT == *itU);
    }
}

/**
 * Tests whether a Point2DList has the same elements in the same iteration
 *         order as another container.
 *
 * @copydoc checkElements(T const &, U const &)
 */
// TODO: remove this specialization once Point2DList::const_iterator implemented
template <class U>
void checkElements(Point2DList const &list, U const &u) {
    BOOST_TEST_REQUIRE(list.size() == u.size());
    Point2DList::size_type i;
    typename U::const_iterator itU;
    for (i = 0, itU = begin(u); i < list.size() && itU != end(u); ++i, ++itU) {
        BOOST_TEST(list[i] == *itU);
    }
}

/**
 * Tests whether a Point2DList based on a specific container has
 *        the right values.
 *
 * @tparam C The container type being tested. Must have standard `begin`
 *           and `end` methods or overloads to `std::begin` and `std::end`.
 *           Element type must be convertible to `Point2D`.
 *
 * @param container The container to test.
 */
template <class C>
void checkFromContainer(C const &container) {
    // Make iterators non-const to test for lack of side effects
    C nonConst(container);
    const auto originalOrder = list<typename C::value_type>(begin(nonConst), end(nonConst));
    const auto translatedOrder = list<Point2D>(begin(nonConst), end(nonConst));
    Point2DList testbed = Point2DList(begin(nonConst), end(nonConst));

    // Has the container been modified?
    checkElements(nonConst, originalOrder);

    // Is the new list correct?
    checkElements(testbed, translatedOrder);
}

/// Tests whether the container constructor creates lists with the correct value.
BOOST_AUTO_TEST_CASE(Point2DListIterator, *boost::unit_test::disabled()) {
    using EVector = Point2D::EigenVector;

    initializer_list<Point2D> values = {Point2D(-2.0, 3.5), Point2D(0.0, 0.0), Point2D(15.6, 0.0)};
    initializer_list<EVector> nonPointValues = {EVector(-2.0, 3.5), EVector(0.0, 0.0), EVector(15.6, 0.0)};

    checkFromContainer(vector<Point2D>(values));
    auto vecCompare = [](EVector const &x, EVector const &y) {
        if (x[0] < y[0]) {
            return true;
        } else if (x[1] < y[1]) {
            return true;
        } else {
            return false;
        }
    };
    checkFromContainer(set<EVector, decltype(vecCompare)>(nonPointValues, vecCompare));
    // TODO: find a way to test C-style arrays without code duplication
    // TODO: find a way to test with input-only iterators
}

/**
 * Tests whether the copy-constructor produces identical but
 *        independent objects.
 */
BOOST_AUTO_TEST_CASE(Point2DListCopy) {
    initializer_list<Point2D> values = {Point2D(-2.0, 3.5), Point2D(0.0, 0.0), Point2D(15.6, 0.0)};

    auto original = Point2DList(values);
    auto copy = original;
    BOOST_TEST(original == copy);

    auto extra = Point2D(34.5, -21.2);
    original.push_back(extra);
    copy.pop_back();
    BOOST_TEST(original != copy);
    BOOST_TEST(original.size() == 4u);
    BOOST_TEST(original[0] == values.begin()[0]);
    BOOST_TEST(original[1] == values.begin()[1]);
    BOOST_TEST(original[2] == values.begin()[2]);
    BOOST_TEST(original[3] == extra);
    BOOST_TEST(copy.size() == 2u);
    BOOST_TEST(copy[0] == values.begin()[0]);
    BOOST_TEST(copy[1] == values.begin()[1]);
}

/// Tests whether the move-constructor produces an identical object.
BOOST_AUTO_TEST_CASE(Point2DListMove) {
    initializer_list<Point2D> values = {Point2D(-2.0, 3.5), Point2D(0.0, 0.0), Point2D(15.6, 0.0)};

    auto original = Point2DList(values);
    auto xfer = move(original);
    BOOST_TEST(original != xfer);

    BOOST_TEST(xfer.size() == 3u);
    BOOST_TEST(xfer[0] == values.begin()[0]);
    BOOST_TEST(xfer[1] == values.begin()[1]);
    BOOST_TEST(xfer[2] == values.begin()[2]);
}

/// Tests whether Point2DList::empty() produces the correct output.
BOOST_AUTO_TEST_CASE(empty) {
    Point2DList testbed;
    BOOST_TEST(testbed.empty());

    testbed.push_back(Point2D(34.5, -21.2));
    BOOST_TEST(!testbed.empty());

    testbed.pop_back();
    BOOST_TEST(testbed.empty());
}

/// Tests whether Point2DList::size() produces the correct output.
BOOST_AUTO_TEST_CASE(size) {
    Point2DList testbed;
    BOOST_TEST(testbed.size() == 0u);

    testbed.push_back(Point2D(34.5, -21.2));
    BOOST_TEST(testbed.size() == 1u);

    auto nCopies = testbed.capacity() + 12u;
    // TODO: use insert once it and Point2DList::const_iterator have been implemented
    // testbed.insert(testbed.end(), nCopies, Point2D(3.4, 7.8));
    auto filler = Point2D(3.4, 7.8);
    for (size_t i = 0; i < nCopies; ++i) {
        testbed.push_back(filler);
    }
    BOOST_TEST(testbed.size() == 1u + nCopies);
    BOOST_TEST(testbed.size() <= testbed.capacity());

    testbed.pop_back();
    BOOST_TEST(testbed.size() == nCopies);
}

// TODO: develop a black box test case for Point2DList::max_size()

// TODO: develop a test for operator[] const that distinguishes between a reference and a copy

/// Tests whether Point2DList::push_back() behaves correctly.
BOOST_FIXTURE_TEST_CASE(push_back, Data) {
    for (const std::vector<Point2D> points : lists) {
        Point2DList testbed(points.begin(), points.end());
        const Point2D copyable(42.0, 27.0);
        testbed.push_back(copyable);
        testbed.push_back(std::move(Point2D(-8.0, -27.0)));

        BOOST_REQUIRE(testbed.size() == points.size() + 2);
        for (size_t i = 0; i < points.size(); ++i) {
            BOOST_TEST(points[i] == testbed[i]);
        }
        BOOST_TEST(testbed[points.size()] == copyable);
        BOOST_TEST(testbed[points.size() + 1] == Point2D(-8.0, -27.0));
    }
    checkMaxMethod([](Point2DList &list) { list.push_back(Point2D()); });
}

/// Tests whether Point2DList::emplace_back() behaves correctly.
BOOST_FIXTURE_TEST_CASE(emplace_back, Data) {
    for (const std::vector<Point2D> points : lists) {
        Point2DList testbed(points.begin(), points.end());
        const Point2D copyable(42.0, 27.0);
        testbed.emplace_back(copyable);
        testbed.emplace_back(-8.0, -27.0);

        BOOST_REQUIRE(testbed.size() == points.size() + 2);
        for (size_t i = 0; i < points.size(); ++i) {
            BOOST_TEST(points[i] == testbed[i]);
        }
        BOOST_TEST(testbed[points.size()] == copyable);
        BOOST_TEST(testbed[points.size() + 1] == Point2D(-8.0, -27.0));
    }
    checkMaxMethod([](Point2DList &list) { list.emplace_back(); });
}

/// Tests whether Point2DList::pop_back() behaves correctly.
BOOST_FIXTURE_TEST_CASE(pop_back, Data) {
    for (const std::vector<Point2D> points : lists) {
        Point2DList testbed(points.begin(), points.end());
        if (testbed.size() > 0) {
            testbed.pop_back();

            BOOST_REQUIRE(testbed.size() == points.size() - 1);
            for (size_t i = 0; i < testbed.size(); ++i) {
                BOOST_TEST(points[i] == testbed[i]);
            }
        } else {
            BOOST_CHECK_THROW(testbed.pop_back(), pexExcept::DomainError);
        }
    }
}

/// Tests whether Point2DList::swap() switches list contents.
BOOST_FIXTURE_TEST_CASE(swapElements, Data) {
    for (const std::vector<Point2D> list1 : lists) {
        for (const std::vector<Point2D> list2 : lists) {
            if (list1 == list2) {
                continue;
            }

            Point2DList testbed1(list1.begin(), list1.end());
            Point2DList testbed2(list2.begin(), list2.end());
            testbed1.swap(testbed2);

            BOOST_REQUIRE(testbed1.size() == list2.size());
            BOOST_REQUIRE(testbed2.size() == list1.size());
            for (size_t i = 0; i < list2.size(); ++i) {
                BOOST_TEST(testbed1[i] == list2[i]);
            }
            for (size_t i = 0; i < list1.size(); ++i) {
                BOOST_TEST(testbed2[i] == list1[i]);
            }
        }
    }
}

// TODO: add swap() test for iterators, pointers, references

/// Tests whether Point2DList::operator=(Point2DList const &) behaves correctly.
BOOST_FIXTURE_TEST_CASE(assignCopy, Data) {
    for (const std::vector<Point2D> list1 : lists) {
        for (const std::vector<Point2D> list2 : lists) {
            if (list1 == list2) {
                continue;
            }

            Point2DList testbed1(list1.begin(), list1.end());
            Point2DList testbed2(list2.begin(), list2.end());
            testbed1 = testbed2;

            BOOST_TEST(testbed1 == testbed2);
            // TODO: test that elements not shared between testbed1 and testbed2
        }
    }
}

/// Tests whether Point2DList::operator=(Point2DList &&) behaves correctly.
BOOST_FIXTURE_TEST_CASE(assignMove, Data) {
    for (const std::vector<Point2D> list1 : lists) {
        for (const std::vector<Point2D> list2 : lists) {
            if (list1 == list2) {
                continue;
            }

            Point2DList testbed1(list1.begin(), list1.end());
            Point2DList testbed2(list2.begin(), list2.end());
            // Make a copy to avoid corrupting testbed2
            testbed1 = std::move(Point2DList(testbed2));

            BOOST_TEST(testbed1 == testbed2);
        }
    }
}

/// Tests whether Point2DList::clear() behaves correctly.
BOOST_FIXTURE_TEST_CASE(clear, Data) {
    for (const std::vector<Point2D> points : lists) {
        Point2DList testbed(points.begin(), points.end());
        testbed.clear();

        BOOST_TEST(testbed.empty());
        // TODO: use iterators to confirm no accessible objects
    }
}

/// Tests whether Point2DList::capacity() satisfies invariants.
BOOST_FIXTURE_TEST_CASE(capacity, Data) {
    for (const std::vector<Point2D> points : lists) {
        Point2DList testbed(points.begin(), points.end());

        BOOST_TEST(testbed.capacity() >= testbed.size());
        // Modification tests already done in size
    }
}

/// Tests whether Point2DList::reserve() behaves correctly.
BOOST_FIXTURE_TEST_CASE(reserve, Data) {
    for (const std::vector<Point2D> points : lists) {
        Point2DList testbed(points.begin(), points.end());

        auto oldCapacity = testbed.capacity();
        testbed.reserve(1);
        BOOST_TEST(testbed.capacity() == oldCapacity);
        testbed.reserve(testbed.capacity());
        BOOST_TEST(testbed.capacity() == oldCapacity);
        testbed.reserve(2 * oldCapacity);
        BOOST_TEST(testbed.capacity() >= 2 * oldCapacity);
    }
    checkMaxConstructor(
            [](Point2DList::size_type n) {
                Point2DList x;
                x.reserve(n);
            },
            "reserve(size_type)");
}

// TODO: test cases for both forms of data()

/**
 * Test == for typical properties of equality operators.
 *
 * The == operator is tested for reflexivity and symmetry. Test for
 * transitivity to be added later.
 */
BOOST_FIXTURE_TEST_CASE(equality, Data) {
    for (const std::vector<Point2D> list1 : lists) {
        Point2DList testbed1(list1.begin(), list1.end());

        BOOST_TEST(testbed1 == testbed1);
        BOOST_TEST(!(testbed1 != testbed1));

        Point2DList copy = testbed1;
        BOOST_TEST(testbed1 == copy);
        BOOST_TEST(copy == testbed1);

        for (const std::vector<Point2D> list2 : lists) {
            if (list1 == list2) {
                continue;
            }
            Point2DList testbed2(list2.begin(), list2.end());
            BOOST_TEST(testbed1 != testbed2);
            BOOST_TEST(testbed2 != testbed1);
        }
    }
}

/// Test if == and != give mutually consistent results.
BOOST_FIXTURE_TEST_CASE(inequality, Data) {
    for (const std::vector<Point2D> list1 : lists) {
        Point2DList testbed1(list1.begin(), list1.end());
        for (const std::vector<Point2D> list2 : lists) {
            Point2DList testbed2(list2.begin(), list2.end());
            BOOST_REQUIRE_EQUAL((list1 == list2), (list2 == list1));
            BOOST_REQUIRE_EQUAL((list1 != list2), (list2 != list1));
            BOOST_CHECK_NE((list1 == list2), (list1 != list2));
        }
    }
}
}
}
} /* namespace lsst::afw::geom */
