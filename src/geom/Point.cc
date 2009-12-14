#include "lsst/afw/geom/Point.h"

namespace geom = lsst::afw::geom;

template <typename T, int N>
geom::CoordinateExpr<N> geom::Point<T,N>::operator<(Point const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] < other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::Point<T,N>::operator<=(Point const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] <= other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::Point<T,N>::operator>(Point const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] > other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::Point<T,N>::operator>=(Point const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] >= other[n];
    return r;
}
