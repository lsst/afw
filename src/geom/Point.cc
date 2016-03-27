/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
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
    for (int n=0; n<N; ++n) {
        this->_vector[n] = detail::PointSpecialized<T>::template convert<U>(other[n]);
    }
}

template <typename T>
template <typename U>
geom::Point<T,2>::Point(Point<U,2> const & other) : Super() {
    for (int n=0; n<2; ++n) {
        this->_vector[n] = detail::PointSpecialized<T>::template convert<U>(other[n]);
    }
}

template <typename T>
template <typename U>
geom::Point<T,3>::Point(Point<U,3> const & other) : Super() {
    for (int n=0; n<3; ++n) {
        this->_vector[n] = detail::PointSpecialized<T>::template convert<U>(other[n]);
    }
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::PointBase<T,N>::eq(Point<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (int n=0; n<N; ++n) r[n] = this->_vector[n] == other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::PointBase<T,N>::ne(Point<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (int n=0; n<N; ++n) r[n] = this->_vector[n] != other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::PointBase<T,N>::lt(Point<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (int n=0; n<N; ++n) r[n] = this->_vector[n] < other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::PointBase<T,N>::le(Point<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (int n=0; n<N; ++n) r[n] = this->_vector[n] <= other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::PointBase<T,N>::gt(Point<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (int n=0; n<N; ++n) r[n] = this->_vector[n] > other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::PointBase<T,N>::ge(Point<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (int n=0; n<N; ++n) r[n] = this->_vector[n] >= other[n];
    return r;
}

#ifndef DOXYGEN
template class geom::PointBase<int,2>;
template class geom::PointBase<int,3>;
template class geom::PointBase<double,2>;
template class geom::PointBase<double,3>;
template class geom::Point<int,2>;
template class geom::Point<int,3>;
template class geom::Point<double,2>;
template class geom::Point<double,3>;
template geom::Point<int,2>::Point(geom::Point<double,2> const &);
template geom::Point<int,3>::Point(geom::Point<double,3> const &);
template geom::Point<double,2>::Point(geom::Point<int,2> const &);
template geom::Point<double,3>::Point(geom::Point<int,3> const &);
#endif
