/*
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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
#include <type_traits>

#include "boost/tr1/functional.hpp"

#include "ndarray.h"

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Box.h"

#if !defined(LSST_DETECTION_FOOTPRINT_ARRAY_H)
#define LSST_DETECTION_FOOTPRINT_ARRAY_H

namespace lsst{
namespace afw {
namespace detection {
class Footprint;


/**
 * @brief Flatten the first two dimensions of an array
 *
 * Use this footprint to map 2-D points in the source to 1-D position in
 * the destination. This forces a deep copy of the relevant parts of the
 * source.
 *
 * @param[in]  fp      footprint to operate on
 * @param[in]  src     array to copy from.  The first two dimensions are (height, width).
 * @param[out] dest    array to copy to. The dimensions of the dest must be area of
 *                     the footprint, inner N-1 dimensions of the source
 * @param[in]  xy0     xy0 of the src array in the footprint's coordinate system
 *
 * For example,
 \code
 flattenArray(foot, image.getArray(), array, image.getXY0());
 \endcode
 */
template <typename T, typename U, int N, int C, int D>
void flattenArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    ndarray::Array<U, N-1, D> const & dest,
    lsst::afw::geom::Point2I const & xy0 = lsst::afw::geom::Point2I()
);

/**
 * @brief Flatten the first two dimensions of an array
 *
 * Use this footprint to map 2-D points in the source to 1-D position in
 * the destination. This forces a deep copy of the relevant parts of the
 * source.
 *
 * @param[in]  fp      footprint to operate on
 * @param[in]  src     array to copy from.  The first two dimensions are (height, width).
 * @param[out] dest    array to copy to. The dimensions of the dest must be area of
 *                     the footprint, inner N-1 dimensions of the source
 * @param[in]  pixelOp Functor taking src's pixel value, and returning the value of dest
 * @param[in]  xy0     xy0 of the src array in the footprint's coordinate system
 *
 * For example,
 \code
 flattenArray(foot, image.getArray(), array, pixelOp(), image.getXY0());
 \endcode
 */
template <typename T, typename U, int N, int C, int D, typename PixelOpT>
void flattenArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    ndarray::Array<U, N-1, D> const & dest,
    PixelOpT const& pixelOp,
    lsst::afw::geom::Point2I const & xy0 = lsst::afw::geom::Point2I()
);

/**
 * @brief Flatten the first two dimensions of an array
 * Use this footprint to map 2-D points in the source to 1-D locations in
 * the destination. This forces a deep copy of some of the relevant parts of
 * source.
 *
 * @param[in]  fp      footprint to operate on
 * @param[in]  src     array to copy from.  The first two dimensions are (height, width);
 *                     the remainder are copied exactly.
 * @param[in]  xy0  xy0 of the src array in the footprint's coordinate system
 *
 * For example,
 \code
 array = flattenArray(foot, image.getArray(), image.getXY0());
 \endcode
 */
template <typename T, int N, int C>
ndarray::Array<typename std::remove_const<T>::type, N-1, N-1> flattenArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    lsst::afw::geom::Point2I const & xy0 = lsst::afw::geom::Point2I()
);

/**
 * @brief expand the first dimension of an array
 *
 * Use this footprint to map 1-D positions in the source to 2-D points in
 * the destination. This forces a deep copy of the relevant parts of the
 * source.
 *
 * @param[in]  fp      footprint to operate on
 * @param[in]  src     array to copy from. The size of the outer dimension must match
 *                     the area of the footprint.
 * @param[out] dest    array to copy to. The dimensions of the array must be height,
 *                     width, inner N-1 dimensions of the source.
 * @param[in]  xy0  xy0 of the src array in the footprint's coordinate system
 *
 * For example,
 \code
 expandArray(foot, array, image.getArray(), image.getXY0());
 \endcode
 */
template <typename T, typename U, int N, int C, int D>
void expandArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    ndarray::Array<U, N+1, D> const & dest,
    lsst::afw::geom::Point2I const & xy0 = lsst::afw::geom::Point2I()
);

/**
 * @brief expand the first dimension of an array, applying a functor to each pixel
 *
 * Use this footprint to map 1-D positions in the source to 2-D points in
 * the destination. This forces a deep copy of the relevant parts of the
 * source.
 *
 * @param[in]  fp      footprint to operate on
 * @param[in]  src     array to copy from. The size of the outer dimension must match
 *                     the area of the footprint.
 * @param[out] dest    array to copy to. The dimensions of the array must be height,
 *                     width, inner N-1 dimensions of the source.
 * @param[in]  pixelOp Functor taking src's pixel value, and returning the value of dest
 * @param[in]  xy0  xy0 of the src array in the footprint's coordinate system
 */
template <typename T, typename U, int N, int C, int D, typename PixelOpT>
void expandArray(
    Footprint const & fp,
    ndarray::Array<T, N, C> const & src,
    ndarray::Array<U, N+1, D> const & dest,
    PixelOpT const& pixelOp,
    lsst::afw::geom::Point2I const & xy0 = lsst::afw::geom::Point2I()
);

/**
 * @brief expand the first dimension of an array
 *
 * Use this footprint to map 1-D positions in the source to 2-D points in
 * the destination. This whose first two dimension are determined by the
 * bounds of this footprint, and whose remaming dimensions are determined by
 * the inner N-1 dimensions of the source. the forces a deep copy of the source
 *
 * @param[in]  fp      footprint to operate on
 * @param[in]  src     array to copy from. The size of the outer dimension must match
 *                     the area of the footprint.
 * @param[in]  bbox    bounding box of the returned array.
 */
template <typename T, int N, int C>
ndarray::Array<typename std::remove_const<T>::type, N+1, N+1> expandArray(
    Footprint const & fp,
    ndarray::Array<T, N, C> const & src,
    lsst::afw::geom::Box2I const & bbox = lsst::afw::geom::Box2I()
);

}}}

#endif


