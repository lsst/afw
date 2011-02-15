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

#include "lsst/afw/detection/FootprintArray.h"

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
 * @param src array to copy from. Size of outer dimensions must be match 
 *            the height and width of the bounds of this footprint.
 * @param dest array to copy to. The dimensions of the dest must be area of
 *             the footprint, inner N-1 dimensions of the source
 */     
template <typename T, typename U, int N, int C, int D>
void flattenArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    ndarray::Array<typename boost::remove_const<U>::type, N-1, D> const & dest
) {
    typedef T ConstT;
    typedef typename boost::remove_const<U>::type MutableT;
    typedef ndarray::Array<ConstT, N, C> SourceT; 
    typedef ndarray::Array<MutableT, N-1, D> DestT; 

    geom::BoxI box = fp.getBBox();
    if (src.template getSize<0>() != box.getHeight() || 
        src.template getSize<1>() != box.getWidth()
    ) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Destination array has outer sizes (%d,%d) not matching"
                           "footprint bounds (%d,%d)"
                           )%src.template getSize<0>()%src.template getSize<1>()
                            %box.getHeight()%box.getWidth()
            ).str()
        );
    }
    if(dest.template getSize<0>() != fp.getArea()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Source array has outer size %d not matching"
                           "footprint area %d"
                           )%dest.template getSize<0>()%fp.getArea()
            ).str()
        );
    }

    int minX = box.getMinX();
    int minY = box.getMinY();


    typename DestT::Iterator destIter(dest.begin());
    for(Footprint::SpanList::const_iterator s=fp.getSpans().begin(); 
        s != fp.getSpans().end(); ++s
    ) {
        Span const & span = **s;
        typename SourceT::Reference row(src[span.getY()-minY]);
        copy(row.begin()+span.getX0() - minX, row.begin()+span.getWidth(), destIter);
        destIter+= span.getWidth();
    }
}

/**
 * @brief Flatten the first two dimensions of an array
 * Use this footprint to map 2-D points in the source to 1-D locations in
 * the destination. This forces a deep copy of some of the relevant parts of
 * source.
 *
* @param src array to copy from. Size of outer dimensions must be match 
 *            the height and width of the bounds of this footprint.
 * @param dest array to copy to. The dimensions of the dest will be area of
 *            the footprint, N-1 dimensions of the source
 */     
template <typename T, int N, int C>
ndarray::Array<typename boost::remove_const<T>::type, N-1, N-1> flattenArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src
) {
    ndarray::Array<typename boost::remove_const<T>::type, N-1,N-1> dest = ndarray::allocate(
        ndarray::concatenate(fp.getArea(), src.template getShape().template last<N-1>())
    );
    flattenArray(fp, src, dest);
    return dest;
}

/**
 * @brief expand the first dimension of an array
 *
 * Use this footprint to map 1-D positions in the source to 2-D points in
 * the destination. This forces a deep copy of the relevant parts of the
 * source.
 *
 * @param src array to copy from. The size of the outer dimension must match
 *            the area of the footprint
 * @param dest array to copy to. The dimensions of the array must be height,
 *             width, inner N-1 dimensions of the source
 */
template <typename T, typename U, int N, int C, int D>
void expandArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    ndarray::Array<typename boost::remove_const<U>::type, N+1, D> const & dest
) {
    typedef T ConstT;
    typedef typename boost::remove_const<U>::type MutableT;
    typedef ndarray::Array<ConstT, N, C> SourceT; 
    typedef ndarray::Array<MutableT, N+1, D> DestT; 

    geom::BoxI box(fp.getBBox());
    if(dest.template getSize<0>() != box.getHeight() || 
       dest.template getSize<1>() != box.getWidth()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Destination array has outer sizes (%d,%d) not matching"
                           "footprint bounds (%d,%d)"
                           )%dest.template getSize<0>()%dest.template getSize<1>()
                            %box.getHeight()%box.getWidth()
            ).str()
        );
    }
    if(src.template getSize<0>() != fp.getArea()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Source array has outer size %d not matching"
                           "footprint area %d"
                           )%src.template getSize<0>()%fp.getArea()
            ).str()
        );
    }
    int minX = box.getMinX();
    int minY = box.getMinY();

    typename SourceT::Iterator srcIter(src.begin());
    for(Footprint::SpanList::const_iterator s=fp.getSpans().begin(); 
        s != fp.getSpans().end(); ++s
    ) {
        Span const & span = **s;
        typename DestT::Reference row(dest[span.getY()-minY]);
        copy(srcIter, srcIter+span.getWidth(), row.begin()+span.getX0());
        srcIter+= span.getWidth();
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
 * @param src array to copy from. The size of the outer dimension must match
 *            the area of the footprint
 * @return a deep copy of the source. The dimensions of the array will
 *            height, width, inner N-1 dimensions of the source.
 */
template <typename T, int N, int C>
ndarray::Array<typename boost::remove_const<T>::type, N+1, N+1> expandArray(
    Footprint const & fp,
    ndarray::Array<T, N, C> const & src
) {    
    geom::BoxI box(fp.getBBox());
    ndarray::Array<typename boost::remove_const<T>::type, N+1, N+1> dest = ndarray::allocate(
        ndarray::concatenate(
            ndarray::makeVector(box.getHeight(), box.getWidth()), 
            src.template getShape().template last<N-1>()
        )
    );
    expandArray(fp, src, dest);
    return dest;
}

}}}


