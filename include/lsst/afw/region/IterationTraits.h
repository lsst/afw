// -*- lsst-c++ -*-
#ifndef LSST_AFW_REGION_IterationTraits_INCLUDED
#define LSST_AFW_REGION_IterationTraits_INCLUDED

/**
 *  @file IterationTraits.h
 *
 *  When we pass an array as a target argument to Region::apply, we need to know whether
 *  to iterate over it as if it's an image (i.e. the first two dimensions are (y, x)),
 *  or whether to treat it as a flattened array (i.e. the first dimension corresponds to the
 *  area of the Region).  It's not enough just to test the dimension of the array - we
 *  may want to pass in, for instance, a matrix with dimensions (pixels, model parameters);
 *  in that case, we'd want flattened iteration, and to pass the entire column that corresponds
 *  to each pixel as the target argument of onPixel.
 *
 *  By specializing IterationTraits for a particular class, we tell Region::apply how
 *  to iterate over it; we could do this for the image classes, for instance (and we do, 
 *  but not here, because the pixels subpackage doesn't depend on the image subpackage).
 *
 *  For ndarray::Arrays, we don't pass them in directly as targets; instead we wrap them
 *  with asImage() or asFlat(), which return proxy classes that do have IterationTraits
 *  specializations.
 */
#include "boost/ref.hpp"
#include "lsst/ndarray.hpp"

namespace lsst { namespace afw { namespace region {

template <typename Target> struct IterationTraits;

template <typename U>
struct IterationTraits< boost::reference_wrapper<U> > : public IterationTraits<U> {};

template <typename T, int N, int C>
struct ArrayImageProxy {
    ndarray::Array<T,N,C> array;

    explicit ArrayImageProxy(ndarray::Array<T,N,C> const & array) : array(array_) {}
};

template <typename T, int N, int C>
struct ArrayOffsetImageProxy {
    ndarray::Array<T,N,C> array;
    geom::Extent2i offset;

    explicit ArrayOffsetImageProxy(ndarray::Array<T,N,C> const & array, geom::Extent2I const & offset_) : 
        array(array_), offset(offset_)
    {}
};

template <typename T, int N, int C>
struct ArrayFlatProxy {
    ndarray::Array<T,N,C> array;

    explicit ArrayFlatProxy(ndarray::Array<T,N,C> const & array) : array(array_) {}
};

template <typename T, int N, int C>
ArrayImageProxy<T,N,C> asImage(ndarray::Array<T,N,C> const & array) {
    return ArrayImageProxy<T,N,C>(array);
}

template <typename T, int N, int C>
ArrayOffsetImageProxy<T,N,C> asImage(ndarray::Array<T,N,C> const & array, geom::Extent2I const & offset) {
    return ArrayOffsetImageProxy<T,N,C>(array, offset);
}

template <typename T, int N, int C>
ArrayFlatProxy<T,N,C> asFlat(ndarray::Array<T,N,C> const & array) {
    return ArrayFlatProxy<T,N,C>(array);
}

template <typename T, int N, int C>
struct IterationTraits< ArrayImageProxy<T,N,C> > {

    typedef typename Array<T,N,C>::Reference::Iterator Iterator;
    
    static Iterator getIterator(ArrayImageProxy<T,N,C> const & target, int n, Span const & span) {
        return target.array[span.getY()].begin() + span.getX0();
    }

};

template <typename T, int N, int C>
struct IterationTraits< ArrayOffsetImageProxy<T,N,C> > {

    typedef typename Array<T,N,C>::Reference::Iterator Iterator;

    static Iterator getIterator(ArrayOffsetImageProxy<T,N,C> const & target, int n, Span const & span) {
        return target.array[span.getY() + target.offset.getY()].begin() + span.getX0()
            + target.offset.getX();
    }

};

template <typename T, int N, int C>
struct IterationTraits< ArrayFlatProxy<T,N,C> > {

    typedef typename Array<T,N,C>::Iterator Iterator;

    static Iterator getIterator(ArrayFlatProxy<T,N,C> const & target, int n, Span const & span) {
        return target.array.begin() + n;
    }

};

}}} // namespace lsst::afw::pixels

#endif // !LSST_AFW_REGION_IterationTraits_INCLUDED
