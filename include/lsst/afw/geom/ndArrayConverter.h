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

#ifndef LSST_AFW_GEOM_NDARRAYCONVERTER_H_
#define LSST_AFW_GEOM_NDARRAYCONVERTER_H_

//#include <iterator>
#include <type_traits>
#include <vector>

#include "ndarray.h"

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/SpherePoint.h"

namespace lsst {
namespace afw {
namespace geom {

/*
 * Conversion methods for Point.
 */

/**
 * Converts a single Point to a multidimensional array of coordinates.
 *
 * @tparam T,N the template parameters for Point
 *
 * @param point the Point to convert
 *
 * @return an array with dimensions of 1 &times; `N`. `array[0][i]` shall
 *         correspond to `point[i]` for all `i`.
 *
 * @exceptsafe Provides strong exception guarantee.
 */
template <typename T, int N>
ndarray::Array<T, 2, 2> pointToNdArray(Point<T, N> const &point) {
    throw std::runtime_error("Not yet implemented");
}

/**
 * Converts a container of Point to a multidimensional array of coordinates.
 *
 * Any operations on the returned array shall not affect the original container.
 *
 * @tparam ForwardIterator a forward iterator to a container of Points.
 *
 * @param first an iterator pointing to the first element to convert
 * @param last an iterator pointing to just after the last element to convert
 *
 * @return an array with dimensions of `distance(first, last)` &times; `N`,
 *         where `N` is the template parameter to `Point`. The element type
 *         shall match the element type of the Points in the container. For the
 *         `i`th point in the container, `array[i][j]` shall equal `point[j]`
 *         for valid indices `j`.
 *
 * @exceptsafe Provides strong exception guarantee.
 */
// TODO: ensure pybind11 wrapper accepts a list
template <typename ForwardIterator>
ndarray::Array<typename std::iterator_traits<ForwardIterator>::value_type::Element, 2, 2> pointToNdArray(
        ForwardIterator first, ForwardIterator last) {
    throw std::runtime_error("Not yet implemented");
}

/**
 * Converts a multidimensional array of coordinates to a vector of Point.
 *
 * Any operations on the returned vector shall not affect the original array.
 *
 * @tparam T the type of the coordinate data
 * @tparam M the number of guaranteed contiguous rows in `array`; ignored
 *
 * @param array an N &times; 2 array of coordinates. A slice of the form
 *              `array[view(i)()]` must represent the `i`th point.
 *
 * @return a vector of length `array.getSize<0>()`. Each point shall have
 *         coordinates in the same order as in `array`, i.e., if
 *         `vec = ndArrayToPoint2(array)`, then `vec.at(i)[j]` shall equal
 *         `array[i][j]` for valid indices `i` and `j`.
 *
 * @throws InvalidParameterError Thrown if `array` does not have the correct
 *                               dimensions.
 *
 * @exceptsafe Provides strong exception guarantee.
 */
// Template parameter M is needed because C++11 template type deduction
// ignores the implicit conversion of Array<T, 2, X> to Array<T, 2, 0>
template <typename T, int M>
std::vector<Point<T, 2>> ndArrayToPoint2(ndarray::Array<T, 2, M> const &array) {
    throw std::runtime_error("Not yet implemented");
}

/**
 * @copybrief ndArrayToPoint2(ndarray::Array<T, 2, 0> const &)
 *
 * As ndArrayToPoint2(ndarray::Array<T, 2, 0> const &), but converts
 * three-dimensional points.
 */
template <typename T, int M>
std::vector<Point<T, 3>> ndArrayToPoint3(ndarray::Array<T, 2, M> const &array) {
    throw std::runtime_error("Not yet implemented");
}

/*
 * Conversion methods for SpherePoint.
 */

/**
 * Converts a single SpherePoint to a multidimensional array of coordinates.
 *
 * @param point the SpherePoint to convert
 *
 * @return an array with dimensions of 1 &times; 2. `array[0][0]` shall  be the
 *         longitude and `array[0][1]` the latitude, both in radians.
 *
 * @exceptsafe Provides strong exception guarantee.
 */
ndarray::Array<double, 2, 2> spherePointToNdArray(SpherePoint const &point);

/**
 * Converts a container of SpherePoint to a multidimensional array of
 * coordinates.
 *
 * Any operations on the returned array shall not affect the original container.
 *
 * @tparam ForwardIterator a forward iterator to a container of SpherePoints.
 *
 * @param first an iterator pointing to the first element to convert
 * @param last an iterator pointing to just after the last element to convert
 *
 * @return an array with dimensions of `distance(first, last)` &times; 2.
 *         For the `i`th point in the container, `array[i][0]` shall  be the
 *         longitude and `array[i][1]` the latitude, both in radians.
 *
 * @exceptsafe Provides strong exception guarantee.
 */
// TODO: ensure pybind11 wrapper accepts a list
template <typename ForwardIterator>
ndarray::Array<double, 2, 2> spherePointToNdArray(ForwardIterator first, ForwardIterator last) {
    throw std::runtime_error("Not yet implemented");
}

/**
 * Converts a multidimensional array of coordinates to a vector of SpherePoint.
 *
 * As ndArrayToPoint2(ndarray::Array<T, 2, 0> const &), but interprets the
 * input as spherical coordinates in radians. `array[i][0]` shall be
 * interpreted as the longitude and `array[i][1]` as the latitude for valid
 * indices `i`.
 *
 * @throws InvalidParameterError Thrown if any point in `array` has a latitude
 *                               outside the interval [-&pi;/2, &pi;/2].
 */
std::vector<SpherePoint> ndArrayToSpherePoint(ndarray::Array<double, 2, 0> const &array);
}
}
} /* namespace lsst::afw::geom */

#endif /* LSST_AFW_GEOM_NDARRAYCONVERTER_H_ */
