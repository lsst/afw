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
 
#include "lsst/afw/geom/CoordinateBase.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_integral.hpp>
namespace geom = lsst::afw::geom;

template <typename T, int N>
geom::Extent<T,N>::Extent(Point<T,N> const & other) :
    Super(other.asEigen())
{}

// The following two template specializations raise Doxygen warnings and produce no documenation.
// This is a known Doxygen bug: <https://bugzilla.gnome.org/show_bug.cgi?id=406027>
/// \cond DOXYGEN_BUG
template <typename T>
geom::Extent<T,2>::Extent(Point<T,2> const & other) :
    Super(other.asEigen())
{}

template <typename T>
geom::Extent<T,3>::Extent(Point<T,3> const & other) :
    Super(other.asEigen())
{}
/// \endcond

template <typename T, int N>
geom::CoordinateExpr<N> geom::ExtentBase<T,N>::eq(Extent<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (int n=0; n<N; ++n) r[n] = this->_vector[n] == other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::ExtentBase<T,N>::ne(Extent<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (int n=0; n<N; ++n) r[n] = this->_vector[n] != other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::ExtentBase<T,N>::lt(Extent<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (int n=0; n<N; ++n) r[n] = this->_vector[n] < other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::ExtentBase<T,N>::le(Extent<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (int n=0; n<N; ++n) r[n] = this->_vector[n] <= other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::ExtentBase<T,N>::gt(Extent<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (int n=0; n<N; ++n) r[n] = this->_vector[n] > other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::ExtentBase<T,N>::ge(Extent<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (int n=0; n<N; ++n) r[n] = this->_vector[n] >= other[n];
    return r;
}

template <typename T, int N>
geom::Point<T,N> geom::ExtentBase<T,N>::asPoint() const {
    return Point<T,N>(static_cast<Extent<T,N> const &>(*this));
}

template <typename T, int N>
geom::Point<T,N> geom::ExtentBase<T,N>::operator+(Point<T,N> const & other) const {
    return Point<T,N>(this->_vector + other.asEigen());
}

template <int N>
geom::Extent<int,N> geom::truncate(Extent<double,N> const & input) {
    Extent<int,N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = static_cast<int>(input[i]);
    }
    return result;
}

template <int N>
geom::Extent<int,N> geom::floor(Extent<double,N> const & input) {
    Extent<int,N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = std::floor(input[i]);
    }
    return result;
}

template <int N>
geom::Extent<int,N> geom::ceil(Extent<double,N> const & input) {
    Extent<int,N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = std::ceil(input[i]);
    }
    return result;
}

#ifndef DOXYGEN

template class geom::ExtentBase<int,2>;
template class geom::ExtentBase<int,3>;
template class geom::ExtentBase<double,2>;
template class geom::ExtentBase<double,3>;
template class geom::Extent<int,2>;
template class geom::Extent<int,3>;
template class geom::Extent<double,2>;
template class geom::Extent<double,3>;
template geom::Extent<double,2>::Extent(geom::Extent<int,2> const &);
template geom::Extent<double,3>::Extent(geom::Extent<int,3> const &);
template geom::Extent<double,2>::Extent(geom::Point<int,2> const &);
template geom::Extent<double,3>::Extent(geom::Point<int,3> const &);

template geom::Extent<int,2> geom::truncate(geom::Extent<double,2> const &);
template geom::Extent<int,3> geom::truncate(geom::Extent<double,3> const &);
template geom::Extent<int,2> geom::floor(geom::Extent<double,2> const &);
template geom::Extent<int,3> geom::floor(geom::Extent<double,3> const &);
template geom::Extent<int,2> geom::ceil(geom::Extent<double,2> const &);
template geom::Extent<int,3> geom::ceil(geom::Extent<double,3> const &);

#endif // !DOXYGEN
