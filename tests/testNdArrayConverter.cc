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
#define BOOST_TEST_MODULE NdArrayConverterCpp

#include "boost/test/unit_test.hpp"

#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/ndArrayConverter.h"
#include "lsst/pex/exceptions.h"

namespace pexExcept = lsst::pex::exceptions;
using namespace std;

/*
 * Unit tests for *Point* <-> ndarray conversions.
 *
 * Many of these tests can be translated to Python once a Pybind11 interface is available.
 */
namespace lsst {
namespace afw {
namespace geom {

/**
 * Common data for Point2D tests.
 *
 * Boost dataset framework would be better, but for some reason references to
 *     BOOST_DATA_TEST_CASE won't compile.
 */
struct ToNdArray2D {
    ToNdArray2D() : lists() {
        const Point2D filler(5.0, 3.0);
        lists.emplace_back();
        lists.emplace_back(std::initializer_list<Point2D>{filler});
        lists.emplace_back(
                std::initializer_list<Point2D>{Point2D(-3.0, 5.4), Point2D(8.0, 8.0), Point2D(4.5, 6.0)});
        lists.emplace_back(100, filler);
    }

    constexpr static double uniqueValue = 134.0;
    std::list<std::list<Point2D>> lists;
};

/// Common data for Point2D reverse tests.
struct FromNdArray2D {
    FromNdArray2D() : arrays() {
        using namespace ndarray;
        const Point2D filler(5.0, 3.0);

        arrays.emplace_back();

        Array<double, 2, 2> singleton = allocate(1, 2);
        singleton[0][0] = filler[0];
        singleton[0][1] = filler[1];
        arrays.push_back(singleton);

        Array<double, 2, 2> shortList = allocate(3, 2);
        shortList[0][0] = -3.0;
        shortList[0][1] = 5.4;
        shortList[1][0] = 8.0;
        shortList[1][1] = 8.0;
        shortList[2][0] = 4.5;
        shortList[2][1] = 6.0;
        arrays.push_back(shortList);

        Array<double, 2, 2> longList = allocate(100, 2);
        longList[view()(0)] = filler[0];
        longList[view()(1)] = filler[1];
        arrays.push_back(longList);
    }

    constexpr static double uniqueValue = 134.0;
    std::list<ndarray::Array<double, 2, 2>> arrays;
};

/// Common data for Point3I tests.
struct ToNdArray3I {
    ToNdArray3I() : lists() {
        const Point3I filler(5.0, 3.0, 4.0);
        lists.emplace_back();
        lists.emplace_back(std::initializer_list<Point3I>{filler});
        lists.emplace_back(std::initializer_list<Point3I>{Point3I(-3.0, 5.4, 7.0), Point3I(8.0, 8.0, 0.0),
                                                          Point3I(4.5, 6.0, -9.5)});
        lists.emplace_back(100, filler);
    }

    constexpr static int uniqueValue = 134;
    std::list<std::list<Point3I>> lists;
};

/// Common data for Point3I reverse tests.
struct FromNdArray3I {
    FromNdArray3I() : arrays() {
        using namespace ndarray;
        const Point3I filler(5.0, 3.0, 4.0);

        arrays.emplace_back();

        Array<int, 2, 2> singleton = allocate(1, 3);
        singleton[0][0] = filler[0];
        singleton[0][1] = filler[1];
        singleton[0][2] = filler[2];
        arrays.push_back(singleton);

        Array<int, 2, 2> shortList = allocate(3, 3);
        shortList[0][0] = -3.0;
        shortList[0][1] = 5.4;
        shortList[0][2] = 7.0;
        shortList[1][0] = 8.0;
        shortList[1][1] = 8.0;
        shortList[1][2] = 0.0;
        shortList[2][0] = 4.5;
        shortList[2][1] = 6.0;
        shortList[2][2] = -9.5;
        arrays.push_back(shortList);

        Array<int, 2, 2> longList = allocate(100, 3);
        longList[view()(0)] = filler[0];
        longList[view()(1)] = filler[1];
        longList[view()(2)] = filler[2];
        arrays.push_back(longList);
    }

    constexpr static int uniqueValue = 134;
    std::list<ndarray::Array<int, 2, 2>> arrays;
};

/// Common data for SpherePoint tests.
struct ToNdArraySph {
    ToNdArraySph() : lists() {
        const SpherePoint filler(26.0 * degrees, -56.0 * degrees);
        lists.emplace_back();
        lists.emplace_back(std::initializer_list<SpherePoint>{filler});
        lists.emplace_back(std::initializer_list<SpherePoint>{filler,
                                                              SpherePoint(360.0 * degrees, -90.0 * degrees),
                                                              SpherePoint(0.0 * degrees, 0.0 * degrees)});
        lists.emplace_back(100, filler);
    }

    const static Angle uniqueValue;
    std::list<std::list<SpherePoint>> lists;
};
const Angle ToNdArraySph::uniqueValue = 134.0 * degrees;

/// Common data for SpherePoint reverse tests.
struct FromNdArraySph {
    FromNdArraySph() : arrays() {
        using namespace ndarray;
        const SpherePoint filler(26.0 * degrees, -56.0 * degrees);

        arrays.emplace_back();

        // Implicit conversion of Angle to double represents angle in radians
        Array<double, 2, 2> singleton = allocate(1, 2);
        singleton[0][0] = filler[0];
        singleton[0][1] = filler[1];
        arrays.push_back(singleton);

        Array<double, 2, 2> shortList = allocate(3, 2);
        shortList[0][0] = filler[0];
        shortList[0][1] = filler[1];
        shortList[1][0] = 360.0 * degrees;
        shortList[1][1] = -90.0 * degrees;
        shortList[2][0] = 0.0 * degrees;
        shortList[2][1] = 0.0 * degrees;
        arrays.push_back(shortList);

        Array<double, 2, 2> longList = allocate(100, 2);
        longList[view()(0)] = filler[0];
        longList[view()(1)] = filler[1];
        arrays.push_back(longList);
    }

    const static Angle uniqueValue;
    std::list<ndarray::Array<double, 2, 2>> arrays;
};
const Angle FromNdArraySph::uniqueValue = 134.0 * degrees;

/**
 * Test whether pointToNdArray(Point<T, N> const &) creates an identical but
 * independent array.
 */
template <int Dim, class P, typename T>
void testPointtoNdArraySingle(std::list<std::list<P>> const& lists, T uniqueValue) {
    for (auto points : lists) {
        for (const auto point : points) {
            const auto array = pointToNdArray(point);
            const auto dims = array.getShape();
            BOOST_REQUIRE(dims[0] == 1);
            BOOST_REQUIRE(dims[1] == Dim);
            for (size_t i = 0; i < dims[1]; ++i) {
                BOOST_TEST(array[0][i] == point[i]);
            }

            array[0][1] = uniqueValue;
            BOOST_REQUIRE(array[0][1] == uniqueValue);
            BOOST_TEST(array[0][1] != point[1]);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(pointToArray2DSingle, ToNdArray2D) {
    testPointtoNdArraySingle<2>(lists, uniqueValue);
}

BOOST_FIXTURE_TEST_CASE(pointToArray3ISingle, ToNdArray3I) {
    testPointtoNdArraySingle<3>(lists, uniqueValue);
}

/**
 * Test whether pointToNdArray(ForwardIterator, ForwardIterator) creates an
 * identical but independent array.
 */
template <int Dim, class C, class T>
void testPointtoNdArrayMulti(C const& collection, T uniqueValue) {
    const auto array = pointToNdArray(collection.begin(), collection.end());
    const auto dims = array.getShape();
    BOOST_REQUIRE(dims[0] == collection.size());
    BOOST_REQUIRE(dims[1] == Dim);

    size_t i;
    typename C::const_iterator it;
    for (i = 0, it = collection.begin(); i < dims[0] && it != collection.end(); ++i, ++it) {
        for (size_t j = 0; j < dims[1]; ++j) {
            BOOST_TEST(array[i][j] == (*it)[j]);
        }
    }

    if (collection.size() > 0) {
        array[0][1] = uniqueValue;
        BOOST_REQUIRE(array[0][1] == uniqueValue);
        BOOST_TEST(array[0][1] != (*collection.begin())[1]);
    }
}

/// Define a strict weak ordering on Point<T,N>
template <typename T, int N>
bool pointSort(Point<T, N> const& lhs, Point<T, N> const& rhs) {
    for (size_t i = 0; i < N; ++i) {
        if (lhs[i] < rhs[i])
            return true;
        else if (lhs[i] > rhs[i])
            return false;
    }
    return false;
}

BOOST_FIXTURE_TEST_CASE(pointToArray2DMulti, ToNdArray2D) {
    for (auto testList : lists) {
        const auto sort = &pointSort<double, 2>;
        const auto testSet = std::set<Point2D, decltype(sort)>(testList.begin(), testList.end(), sort);
        const auto testVector = std::vector<Point2D>(testList.begin(), testList.end());

        testPointtoNdArrayMulti<2>(testList, uniqueValue);
        testPointtoNdArrayMulti<2>(testSet, uniqueValue);
        testPointtoNdArrayMulti<2>(testVector, uniqueValue);
    }
}

BOOST_FIXTURE_TEST_CASE(pointToArray3IMulti, ToNdArray3I) {
    for (auto testList : lists) {
        const auto sort = &pointSort<int, 3>;
        const auto testSet = std::set<Point3I, decltype(sort)>(testList.begin(), testList.end(), sort);
        const auto testVector = std::vector<Point3I>(testList.begin(), testList.end());

        testPointtoNdArrayMulti<3>(testList, uniqueValue);
        testPointtoNdArrayMulti<3>(testSet, uniqueValue);
        testPointtoNdArrayMulti<3>(testVector, uniqueValue);
    }
}

/**
 * Test whether ndArrayToPoint2(Array<T, 2, 0> const &) creates an identical
 * but independent vector.
 */
BOOST_FIXTURE_TEST_CASE(arrayToPoint2Value, FromNdArray2D) {
    for (ndarray::Array<double, 2, 2> data : arrays) {
        std::vector<Point2D> vec = ndArrayToPoint2(data);
        BOOST_REQUIRE(vec.size() == data.getSize<0>());

        for (size_t i = 0; i < vec.size(); ++i) {
            for (size_t j = 0; j < 2; ++j) {
                BOOST_TEST(data[i][j] == vec[i][j]);
            }
        }

        if (vec.size() > 0) {
            vec[0][1] = uniqueValue;
            BOOST_REQUIRE(vec[0][1] == uniqueValue);
            BOOST_TEST(vec[0][1] != data[0][1]);
        }
    }
}

/**
 * Test whether ndArrayToPoint2(Array<T, 2, 0> const &) accepts only arrays
 * of width 2.
 */
BOOST_AUTO_TEST_CASE(arrayToPoint2ErrorCheck) {
    using namespace ndarray;
    const Array<double, 2, 2> empty = allocate(0, 2);
    BOOST_CHECK_NO_THROW(ndArrayToPoint2(empty));
    const Array<double, 2, 2> null = allocate(10, 0);
    BOOST_CHECK_THROW(ndArrayToPoint2(null), pexExcept::InvalidParameterError);
    const Array<double, 2, 2> linear = allocate(10, 1);
    BOOST_CHECK_THROW(ndArrayToPoint2(linear), pexExcept::InvalidParameterError);
    const Array<double, 2, 2> threeDee = allocate(10, 3);
    BOOST_CHECK_THROW(ndArrayToPoint2(threeDee), pexExcept::InvalidParameterError);
}

/**
 * Test whether ndArrayToPoint3(Array<T, 2, 0> const &) creates an identical
 * but independent vector.
 */
BOOST_FIXTURE_TEST_CASE(arrayToPoint3Value, FromNdArray3I) {
    for (ndarray::Array<int, 2, 2> data : arrays) {
        std::vector<Point3I> vec = ndArrayToPoint3(data);
        BOOST_REQUIRE(vec.size() == data.getSize<0>());

        for (size_t i = 0; i < vec.size(); ++i) {
            for (size_t j = 0; j < 3; ++j) {
                BOOST_TEST(data[i][j] == vec[i][j]);
            }
        }

        if (vec.size() > 0) {
            vec[0][1] = uniqueValue;
            BOOST_REQUIRE(vec[0][1] == uniqueValue);
            BOOST_TEST(vec[0][1] != data[0][1]);
        }
    }
}

/**
 * Test whether ndArrayToPoint3(Array<T, 2, 0> const &) accepts only arrays
 * of width 3.
 */
BOOST_AUTO_TEST_CASE(arrayToPoint3ErrorCheck) {
    using namespace ndarray;
    const Array<int, 2, 2> empty = allocate(0, 3);
    BOOST_CHECK_NO_THROW(ndArrayToPoint3(empty));
    const Array<int, 2, 2> null = allocate(10, 0);
    BOOST_CHECK_THROW(ndArrayToPoint3(null), pexExcept::InvalidParameterError);
    const Array<int, 2, 2> linear = allocate(10, 2);
    BOOST_CHECK_THROW(ndArrayToPoint3(linear), pexExcept::InvalidParameterError);
    const Array<int, 2, 2> threeDee = allocate(10, 4);
    BOOST_CHECK_THROW(ndArrayToPoint3(threeDee), pexExcept::InvalidParameterError);
}

/**
 * Test whether spherePointToNdArray(SpherePoint const &) creates an identical
 * but independent array.
 */
BOOST_FIXTURE_TEST_CASE(spherePointToArraySingle, ToNdArraySph) {
    for (auto points : lists) {
        for (const auto point : points) {
            const auto array = spherePointToNdArray(point);
            const auto dims = array.getShape();
            BOOST_REQUIRE(dims[0] == 1);
            BOOST_REQUIRE(dims[1] == 2);
            for (size_t i = 0; i < 2; ++i) {
                BOOST_TEST(array[0][i] == point[i].asRadians());
            }

            array[0][1] = uniqueValue.asRadians();
            BOOST_REQUIRE(array[0][1] == uniqueValue.asRadians());
            BOOST_TEST(array[0][1] != point[1].asRadians());
        }
    }
}

/**
 * Test whether spherePointToNdArray(ForwardIterator, ForwardIterator) creates
 * an identical but independent array.
 */
template <class C>
void testSpherePointtoNdArrayMulti(C const& collection, double uniqueValue) {
    const auto array = spherePointToNdArray(collection.begin(), collection.end());
    const auto dims = array.getShape();
    BOOST_REQUIRE(dims[0] == collection.size());
    BOOST_REQUIRE(dims[1] == 2);

    size_t i;
    typename C::const_iterator it;
    for (i = 0, it = collection.begin(); i < dims[0] && it != collection.end(); ++i, ++it) {
        for (size_t j = 0; j < dims[1]; ++j) {
            BOOST_TEST(array[i][j] == (*it)[j].asRadians());
        }
    }

    if (collection.size() > 0) {
        array[0][1] = uniqueValue;
        BOOST_REQUIRE(array[0][1] == uniqueValue);
        BOOST_TEST(array[0][1] != (*collection.begin())[1]);
    }
}

/// Define a strict weak ordering on SpherePoint
bool sphereSort(SpherePoint const& lhs, SpherePoint const& rhs) {
    if (lhs[0] < rhs[0])
        return true;
    else if (lhs[0] > rhs[0])
        return false;
    else
        return lhs[1] < rhs[1];
}

BOOST_FIXTURE_TEST_CASE(spherePointToArrayMulti, ToNdArraySph) {
    for (auto testList : lists) {
        const auto testSet =
                std::set<SpherePoint, decltype(&sphereSort)>(testList.begin(), testList.end(), &sphereSort);
        const auto testVector = std::vector<SpherePoint>(testList.begin(), testList.end());

        testSpherePointtoNdArrayMulti(testList, uniqueValue);
        testSpherePointtoNdArrayMulti(testSet, uniqueValue);
        testSpherePointtoNdArrayMulti(testVector, uniqueValue);
    }
}

/**
 * Test whether ndArrayToSpherePoint(Array<double, 2, 0> const &) creates an
 * identical but independent vector.
 */
BOOST_FIXTURE_TEST_CASE(arrayToSpherePointValue, FromNdArraySph) {
    for (ndarray::Array<double, 2, 2> data : arrays) {
        std::vector<SpherePoint> vec = ndArrayToSpherePoint(data);
        BOOST_REQUIRE(vec.size() == data.getSize<0>());

        for (size_t i = 0; i < vec.size(); ++i) {
            for (size_t j = 0; j < 2; ++j) {
                BOOST_TEST(data[i][j] == vec[i][j].asRadians());
            }
        }

        if (vec.size() > 0) {
            vec[0][1] = uniqueValue;
            BOOST_REQUIRE(vec[0][1] == uniqueValue);
            BOOST_TEST(vec[0][1].asRadians() != data[0][1]);
        }
    }
}

/// Create a minimal ndarray of SpherePoints that can be used for value testing
ndarray::Array<double, 2, 2> dummySpherePoints() {
    using namespace ndarray;
    const Array<double, 2, 2> points = allocate(2, 2);
    points[0][0] = 30 * degrees;
    points[0][1] = -HALFPI;
    points[1][0] = 172 * degrees;
    points[1][1] = HALFPI;
    return points;
}

/**
 * Test whether ndArrayToSpherePoint(Array<double, 2, 0> const &) accepts only
 * arrays of width 2 with well-defined coordinates.
 */
BOOST_AUTO_TEST_CASE(arrayToSpherePointErrorCheck) {
    using namespace ndarray;
    const Array<double, 2, 2> empty = allocate(0, 2);
    empty.deep() = 0.0;
    BOOST_CHECK_NO_THROW(ndArrayToSpherePoint(empty));
    const Array<double, 2, 2> null = allocate(10, 0);
    null.deep() = 0.0;
    BOOST_CHECK_THROW(ndArrayToSpherePoint(null), pexExcept::InvalidParameterError);
    const Array<double, 2, 2> linear = allocate(10, 1);
    linear.deep() = 0.0;
    BOOST_CHECK_THROW(ndArrayToSpherePoint(linear), pexExcept::InvalidParameterError);
    const Array<double, 2, 2> threeDee = allocate(10, 3);
    threeDee.deep() = 0.0;
    BOOST_CHECK_THROW(ndArrayToSpherePoint(threeDee), pexExcept::InvalidParameterError);

    BOOST_CHECK_NO_THROW(ndArrayToSpherePoint(dummySpherePoints()));
    auto superPole = dummySpherePoints();
    superPole[1][1] = 2.0;
    BOOST_CHECK_THROW(ndArrayToSpherePoint(superPole), pexExcept::InvalidParameterError);
}
}
}
} /* namespace lsst::afw::geom */
