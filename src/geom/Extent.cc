#include "lsst/afw/geom/CoordinateBase.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"

namespace geom = lsst::afw::geom;

template <typename T, int N>
geom::Extent<T,N>::Extent(Point<T,N> const & other) :
    Super(other.asVector())
{}

template <typename T>
geom::Extent<T,2>::Extent(Point<T,2> const & other) :
    Super(other.asVector())
{}

template <typename T>
geom::Extent<T,3>::Extent(Point<T,3> const & other) :
    Super(other.asVector())
{}

template <typename T, int N>
geom::CoordinateExpr<N> geom::ExtentBase<T,N>::eq(Extent<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] == other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::ExtentBase<T,N>::ne(Extent<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] != other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::ExtentBase<T,N>::lt(Extent<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] < other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::ExtentBase<T,N>::le(Extent<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] <= other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::ExtentBase<T,N>::gt(Extent<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] > other[n];
    return r;
}

template <typename T, int N>
geom::CoordinateExpr<N> geom::ExtentBase<T,N>::ge(Extent<T,N> const & other) const {
    CoordinateExpr<N> r;
    for (register int n=0; n<N; ++n) r[n] = this->_vector[n] >= other[n];
    return r;
}

template <typename T, int N>
geom::Point<T,N> geom::ExtentBase<T,N>::operator+(Point<T,N> const & other) const {
    return Point<T,N>(this->_vector + other.asVector());
}

template class geom::ExtentBase<int,2>;
template class geom::ExtentBase<int,3>;
template class geom::ExtentBase<double,2>;
template class geom::ExtentBase<double,3>;
template class geom::Extent<int,2>;
template class geom::Extent<int,3>;
template class geom::Extent<double,2>;
template class geom::Extent<double,3>;
