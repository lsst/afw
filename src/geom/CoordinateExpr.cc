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

namespace lsst { namespace afw { namespace geom {

template <int N>
CoordinateExpr<N> CoordinateExpr<N>::and_(CoordinateExpr<N> const & other) const {
    CoordinateExpr r(*this);
    for (int n=0; n<N; ++n) {
        if (!other[n]) r[n] = false;
    }
    return r;
}

template <int N>
CoordinateExpr<N> CoordinateExpr<N>::or_(CoordinateExpr<N> const & other) const {
    CoordinateExpr r(*this);
    for (int n=0; n<N; ++n) {
        if (other[n]) r[n] = true;
    }
    return r;
}

template <int N>
CoordinateExpr<N> CoordinateExpr<N>::not_() const {
    CoordinateExpr r;
    for (int n=0; n<N; ++n) {
        if (!this->operator[](n)) r[n] = true;
    }
    return r;
}

template class CoordinateExpr<2>;
template class CoordinateExpr<3>;

}}} // end lsst::afw::geom
