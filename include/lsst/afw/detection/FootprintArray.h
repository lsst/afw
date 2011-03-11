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

#include <boost/type_traits.hpp>
#include "lsst/ndarray.h"
#include "lsst/afw/detection/Footprint.h"

#if !defined(LSST_DETECTION_FOOTPRINT_ARRAY_H)
#define LSST_DETECTION_FOOTPRINT_ARRAY_H

namespace lsst{
namespace afw {
namespace detection {

template <typename T, typename U, int N, int C, int D>
void flattenArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    ndarray::Array<U, N-1, D> const & dest,
    geom::Point2I const & origin = geom::Point2I()
);

template <typename T, int N, int C>
ndarray::Array<typename boost::remove_const<T>::type, N-1, N-1> flattenArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    geom::Point2I const & origin = geom::Point2I()
);


template <typename T, typename U, int N, int C, int D>
void expandArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    ndarray::Array<U, N+1, D> const & dest,
    geom::Point2I const & origin = geom::Point2I()
);

template <typename T, int N, int C>
ndarray::Array<typename boost::remove_const<T>::type, N+1, N+1> expandArray(
    Footprint const & fp,
    ndarray::Array<T, N, C> const & src,
    geom::Point2I const & origin = geom::Point2I()
);

}}}

#endif


