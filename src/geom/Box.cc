#include "lsst/afw/geom/Box.h"

namespace geom = lsst::afw::geom;

template <typename T, int N>
geom::Box<T,N>::Box(
    Point<T,N> const & minimum,
    Extent<T,N> const & dimensions
) : _minimum(minimum), _dimensions(dimensions) {}

template class geom::Box<int,2>;
template class geom::Box<int,3>;

template class geom::Box<double,2>;
template class geom::Box<double,3>;
