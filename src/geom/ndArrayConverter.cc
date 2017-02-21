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

#include <lsst/afw/geom/ndArrayConverter.h>

namespace lsst {
namespace afw {
namespace geom {

template <typename T, int N>
ndarray::Array<T, 2, 2> pointToNdArray(Point<T, N> const &point) {
    throw std::runtime_error("Not yet implemented");
}

template <typename T, int M>
std::vector<Point<T, 2>> ndArrayToPoint2(ndarray::Array<T, 2, M> const &array) {
    throw std::runtime_error("Not yet implemented");
}

template <typename T, int M>
std::vector<Point<T, 3>> ndArrayToPoint3(ndarray::Array<T, 2, M> const &array) {
    throw std::runtime_error("Not yet implemented");
}

ndarray::Array<double, 2, 2> spherePointToNdArray(SpherePoint const &point) {
    throw std::runtime_error("Not yet implemented");
}

std::vector<SpherePoint> ndArrayToSpherePoint(ndarray::Array<double, 2, 0> const &array) {
    throw std::runtime_error("Not yet implemented");
}

//
// Explicit instantiations
//
template ndarray::Array<int, 2, 2> pointToNdArray<int, 2>(Point2I const &point);
template ndarray::Array<int, 2, 2> pointToNdArray<int, 3>(Point3I const &point);
template ndarray::Array<double, 2, 2> pointToNdArray<double, 2>(Point2D const &point);
template ndarray::Array<double, 2, 2> pointToNdArray<double, 3>(Point3D const &point);

template std::vector<Point<int, 2>> ndArrayToPoint2<int, 2>(ndarray::Array<int, 2, 2> const &array);
template std::vector<Point<double, 2>> ndArrayToPoint2<double, 2>(ndarray::Array<double, 2, 2> const &array);

template std::vector<Point<int, 3>> ndArrayToPoint3<int, 2>(ndarray::Array<int, 2, 2> const &array);
template std::vector<Point<double, 3>> ndArrayToPoint3<double, 2>(ndarray::Array<double, 2, 2> const &array);

}
}
} /* namespace lsst::afw::geom */
