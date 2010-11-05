/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
#include <cmath>

#include "lsst/afw/geom/Point.h"

namespace geom = lsst::afw::geom;

namespace lsst { namespace afw { namespace geom { namespace detail {

template <typename T>
struct PointSpecialized {};

template <>
struct PointSpecialized<int> {
    template <typename U>
    static int convert(U scalar) {
        return static_cast<int>(std::floor(scalar + 0.5));
    }
};

template <>
struct PointSpecialized<double> {
    template <typename U>
    static double convert(U scalar) {
        return static_cast<double>(scalar);
    }
};

}}}}

template <typename T, int N>
template <typename U>
geom::Point<T,N>::Point(Point<U,N> const & other) : Super() {
    for (register int n=0; n<N; ++n) {
        this->_vector[n] = detail::PointSpecialized<T>::template convert<U>(other[n]);
    }
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::Point<T,N>::eq(Point const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] == other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::Point<T,N>::ne(Point const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] != other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::Point<T,N>::lt(Point const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] < other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::Point<T,N>::le(Point const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] <= other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::Point<T,N>::gt(Point const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] > other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::Point<T,N>::ge(Point const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] >= other[n];
    return r;
}

#ifndef DOXYGEN
template class geom::Point<int,2>;
template class geom::Point<int,3>;
template class geom::Point<double,2>;
template class geom::Point<double,3>;
template geom::Point<int,2>::Point(geom::Point<double,2> const &);
template geom::Point<int,3>::Point(geom::Point<double,3> const &);
template geom::Point<double,2>::Point(geom::Point<int,2> const &);
template geom::Point<double,3>::Point(geom::Point<int,3> const &);
#endif
