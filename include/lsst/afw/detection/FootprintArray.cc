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
#include "lsst/afw/detection/Footprint.h"
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

namespace lsst{
namespace afw{
namespace detection{

namespace {
    /*
     * Check that the Footprint is consistent with the src and dest arrays
     *
     * \note The names @c src and @c dest are appropriate when checking expandArrays's arguments, but
     * are switched when checking flattenArray
     */
    template <typename T, typename U, int N, int C, int D>
    void checkConvertArray(Footprint const & fp,
                          ndarray::Array<T, N, C> const & src,
                          ndarray::Array<U, N+1, D> const & dest,
                          geom::Point2I const & xy0
                         )
    {
        geom::Box2I fpBox = fp.getBBox();
        geom::Box2I imBox(xy0, geom::Extent2I(dest.template getSize<1>(), dest.template getSize<0>()));
        
        if (src.template getSize<0>() != fp.getArea()) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              str(boost::format("Array outer size (%d) does not match"
                                                " footprint area (%d)."
                                               ) % dest.template getSize<0>() % fp.getArea()));
        }

        if (!imBox.contains(fpBox)) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              str(boost::format("Array bounding box (%d) does not contain footprint "
                                                "bounding box (%d)") % imBox % fpBox));
        }
    }
}

template <typename T, typename U, int N, int C, int D>
void flattenArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    ndarray::Array<U, N-1, D> const & dest,
    geom::Point2I const & xy0
) {
    typedef ndarray::Array<T, N, C> SourceT; 
    typedef ndarray::Array<U, N-1, D> DestT; 
    BOOST_STATIC_ASSERT(!boost::is_const<U>::value);

    checkConvertArray(fp, dest, src, xy0);

    typename DestT::Iterator destIter(dest.begin());
    for (Footprint::SpanList::const_iterator s = fp.getSpans().begin(); s != fp.getSpans().end(); ++s) {
        Span const & span = **s;
        typename SourceT::Reference row(src[span.getY() - xy0.getY()]);

        std::copy(
            row.begin() + span.getX0() - xy0.getX(),
            row.begin() + span.getX1() + 1 - xy0.getX(), 
            destIter
        );
        destIter += span.getWidth();
    }
}

template <typename T, typename U, int N, int C, int D, typename PixelOpT>
void flattenArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    ndarray::Array<U, N-1, D> const & dest,
    PixelOpT const& pixelOp,
    geom::Point2I const & xy0
) {
    typedef ndarray::Array<T, N, C> SourceT; 
    typedef ndarray::Array<U, N-1, D> DestT; 
    BOOST_STATIC_ASSERT(!boost::is_const<U>::value);

    checkConvertArray(fp, dest, src, xy0);

    typename DestT::Iterator destIter(dest.begin());
    for (Footprint::SpanList::const_iterator s = fp.getSpans().begin(); s != fp.getSpans().end(); ++s) {
        Span const & span = **s;
        typename SourceT::Reference row(src[span.getY() - xy0.getY()]);
        typename SourceT::Reference::Iterator rowIter = row.begin() + span.getX0() - xy0.getX();
        for (typename DestT::Iterator destEnd = destIter + span.getWidth(); destIter != destEnd;
             ++destIter, ++rowIter) {
            *destIter = *rowIter;
            *rowIter = pixelOp(*rowIter);
        }
    }
}

template <typename T, int N, int C>
ndarray::Array<typename boost::remove_const<T>::type, N-1, N-1> flattenArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    geom::Point2I const & xy0
) {
    ndarray::Vector<int,N-1> shape 
        = ndarray::concatenate(fp.getArea(), src.getShape().template last<N-2>());
    ndarray::Array<typename boost::remove_const<T>::type, N-1,N-1> dest = ndarray::allocate(shape);
    flattenArray(fp, src, dest, xy0);
    return dest;
}

template <typename T, typename U, int N, int C, int D>
void expandArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    ndarray::Array<U, N+1, D> const & dest,
    geom::Point2I const & xy0
                )
{
    typedef ndarray::Array<T, N, C> SourceT; 
    typedef ndarray::Array<U, N+1, D> DestT; 
    BOOST_STATIC_ASSERT(!boost::is_const<U>::value);

    checkConvertArray(fp, src, dest, xy0);

    typename SourceT::Iterator srcIter(src.begin());
    for (Footprint::SpanList::const_iterator s = fp.getSpans().begin(); s != fp.getSpans().end(); ++s) {
        Span const & span = **s;
        typename DestT::Reference row(dest[span.getY() - xy0.getY()]);
        std::copy(srcIter, srcIter + span.getWidth(), row.begin() + span.getX0() - xy0.getX());
        srcIter += span.getWidth();
    }
}

template <typename T, typename U, int N, int C, int D, typename PixelOpT>
void expandArray(
    Footprint const & fp,
    ndarray::Array<T,N,C> const & src,
    ndarray::Array<U, N+1, D> const & dest,
    PixelOpT const& pixelOp,
    geom::Point2I const & xy0
                )
{
    typedef ndarray::Array<T, N, C> SourceT; 
    typedef ndarray::Array<U, N+1, D> DestT; 
    BOOST_STATIC_ASSERT(!boost::is_const<U>::value);

    checkConvertArray(fp, src, dest, xy0);

    typename SourceT::Iterator srcIter(src.begin());
    for (Footprint::SpanList::const_iterator s = fp.getSpans().begin(); s != fp.getSpans().end(); ++s) {
        Span const & span = **s;
        typename DestT::Reference row(dest[span.getY() - xy0.getY()]);

        typename DestT::Reference::Iterator rowIter = row.begin() + span.getX0() - xy0.getX();
        for (typename SourceT::Iterator srcEnd = srcIter + span.getWidth(); srcIter != srcEnd;
             ++srcIter, ++rowIter) {
            *rowIter = pixelOp(*srcIter);
        }
    }
}

template <typename T, int N, int C>
ndarray::Array<typename boost::remove_const<T>::type, N+1, N+1> expandArray(
    Footprint const & fp,
    ndarray::Array<T, N, C> const & src,
    geom::Box2I const & bbox
) {
    geom::Box2I box(bbox);
    if (box.isEmpty()) {
        box = fp.getBBox();
    }
    ndarray::Array<typename boost::remove_const<T>::type, N+1, N+1> dest = ndarray::allocate(
        ndarray::concatenate(
            ndarray::makeVector(box.getHeight(), box.getWidth()), 
            src.getShape().template last<N-1>()
        )
    );
    dest.deep() = 0.0;
    expandArray(fp, src, dest, box.getMin());
    return dest;
}

}}}


