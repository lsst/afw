#include "lsst/afw/geom/CoordinateBase.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"

namespace geom = lsst::afw::geom;

template <typename T, int N>
geom::Extent<T,N>::Extent(Point<T,N> const & other) :
    Super(other.asVector())
{}

template <typename T, int N>
geom::CoordinateExpr<N> geom::Extent<T,N>::operator==(Extent const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] == other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::Extent<T,N>::operator!=(Extent const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] != other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::Extent<T,N>::operator<(Extent const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] < other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::Extent<T,N>::operator<=(Extent const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] <= other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::Extent<T,N>::operator>(Extent const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] > other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::Extent<T,N>::operator>=(Extent const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] >= other[n];
    return r;
}

template <typename T, int N>
geom::Point<T,N> geom::Extent<T,N>::operator+(Point<T,N> const & other) const {
    return Point<T,N>(this->_vector + other.asVector());
}

template class geom::Extent<int,2>;
template class geom::Extent<int,3>;
template class geom::Extent<double,2>;
template class geom::Extent<double,3>;
