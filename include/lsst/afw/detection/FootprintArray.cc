// -*- lsst-c++ -*-

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

/**
 *  @file FootprintArray.cc
 *
 *  @brief Templated source for FootprintArray.h functions.
 *
 *  This is an includeable template source file; it should be included whenever
 *  the functions declared in FootprintArray.h are used.  Note that while
 *  FootprintArray.h is included by afw/detection.h, FootprintArray.cc is not.
 *  
 *  The functions here have too many template parameters for explicit instantiation
 *  to be attractive (because the number of instantiations is combinatorial).
 */

#include "lsst/afw/detection/FootprintArray.h"
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

namespace lsst{
namespace afw{
namespace detection{

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
 * @param[in]  origin  origin of the src array in the footprint's coordinate system
 */
template <typename T, typename U, int N, int C, int D>
void flattenArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    ndarray::Array<U, N-1, D> const & dest,
    geom::Point2I const & origin
) {
    typedef ndarray::Array<T, N, C> SourceT; 
    typedef ndarray::Array<U, N-1, D> DestT; 
    BOOST_STATIC_ASSERT(!boost::is_const<U>::value);

    geom::Box2I fpBox = fp.getBBox();
    geom::Box2I imBox(origin, geom::Extent2I(src.template getSize<1>(), src.template getSize<0>()));

    if (dest.template getSize<0>() != fp.getArea()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Destination array outer size (%d) does not match"
                           " footprint area (%d)."
                           ) % dest.template getSize<0>() % fp.getArea()
            ).str()
        );
    }

    if (!imBox.contains(fpBox)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Source bounding box (%d) does not contain footprint bounding box (%d)")
             % imBox % fpBox).str()
        );
    }

    typename DestT::Iterator destIter(dest.begin());
    for (Footprint::SpanList::const_iterator s = fp.getSpans().begin(); 
         s != fp.getSpans().end(); ++s
    ) {
        Span const & span = **s;
        typename SourceT::Reference row(src[span.getY() - origin.getY()]);
        std::copy(
            row.begin() + span.getX0() - origin.getX(),
            row.begin() + span.getX1() + 1 - origin.getX(), 
            destIter
        );
        destIter += span.getWidth();
    }

}

/**
 * @brief Flatten the first two dimensions of an array
 * Use this footprint to map 2-D points in the source to 1-D locations in
 * the destination. This forces a deep copy of some of the relevant parts of
 * source.
 *
 * @param[in]  fp      footprint to operate on
 * @param[in]  src     array to copy from.  The first two dimensions are (height, width);
 *                     the remainder are copied exactly.
 * @param[in]  origin  origin of the src array in the footprint's coordinate system
 */     
template <typename T, int N, int C>
ndarray::Array<typename boost::remove_const<T>::type, N-1, N-1> flattenArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    geom::Point2I const & origin
) {
    ndarray::Vector<int,N-1> shape 
        = ndarray::concatenate(fp.getArea(), src.template getShape().template last<N-2>());
    std::cerr << "shape: " << shape << "\n";
    ndarray::Array<typename boost::remove_const<T>::type, N-1,N-1> dest = ndarray::allocate(shape);
    flattenArray(fp, src, dest, origin);
    return dest;
}

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
 * @param[in]  origin  origin of the src array in the footprint's coordinate system
 */
template <typename T, typename U, int N, int C, int D>
void expandArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    ndarray::Array<U, N+1, D> const & dest,
    geom::Point2I const & origin
) {
    typedef ndarray::Array<T, N, C> SourceT; 
    typedef ndarray::Array<U, N+1, D> DestT; 
    BOOST_STATIC_ASSERT(!boost::is_const<U>::value);

    geom::Box2I fpBox(fp.getBBox());
    geom::Box2I imBox(origin, geom::Extent2I(dest.template getSize<1>(), dest.template getSize<0>()));

    if (src.template getSize<0>() != fp.getArea()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Source array outer size (%d) does not match"
                           " footprint area (%d)."
                           ) % dest.template getSize<0>() % fp.getArea()
            ).str()
        );
    }

    if (!imBox.contains(fpBox)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Destination bounding box (%d) does not contain footprint bounding box (%d)")
             % imBox % fpBox).str()
        );
    }

    typename SourceT::Iterator srcIter(src.begin());
    for (Footprint::SpanList::const_iterator s = fp.getSpans().begin(); 
        s != fp.getSpans().end(); ++s
    ) {
        Span const & span = **s;
        typename DestT::Reference row(dest[span.getY() - origin.getY()]);
        std::copy(srcIter, srcIter + span.getWidth(), row.begin() + span.getX0() - origin.getX());
        srcIter += span.getWidth();
    }
}

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
ndarray::Array<typename boost::remove_const<T>::type, N+1, N+1> expandArray(
    Footprint const & fp,
    ndarray::Array<T, N, C> const & src,
    geom::Box2I const & bbox
) {
    ndarray::Array<typename boost::remove_const<T>::type, N+1, N+1> dest = ndarray::allocate(
        ndarray::concatenate(
            ndarray::makeVector(bbox.getHeight(), bbox.getWidth()), 
            src.template getShape().template last<N-1>()
        )
    );
    dest.deep() = 0.0;
    expandArray(fp, src, dest, bbox.getMin());
    return dest;
}

}}}


