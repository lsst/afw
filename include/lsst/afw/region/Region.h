// -*- lsst-c++ -*-
#ifndef LSST_AFW_REGION_Region_INCLUDED
#define LSST_AFW_REGION_Region_INCLUDED

#include "boost/ref.hpp"

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/region/Span.h"
#include "lsst/afw/region/IterationTraits.h"

namespace lsst { namespace afw { namespace region {

template <typename Target> class RegionInverse;
template <typename Derived> struct RegionTraits;

/// @brief Enum type returned by region functor's processSpan function.
enum SpanIterationEnum {
    PROCESS_PIXELS, ///< Iterate over all pixels in the span, calling processPixel on each.
    SKIP_SPAN       ///< Do not iterate over pixels in the span.
};


/**
 *  @brief CRTP base class for span-based pixel regions.
 *
 *  A Region is a region of pixels defined by a sequence of non-overlappings Spans.
 *  Unlike the original Footprint class, these spans are not necessarily contiguous, and a
 *  Region has no concept of peaks or an image region.
 *
 *  The Region concept does not define any mutating operations, though an implementation
 *  may have mutators.  A Region should always be normalized - its spans must not overlap, 
 *  and its area and bounding box must be correct.
 */
template <typename Derived>
class Region {
private
    typedef RegionTraits<Derived> Traits;
protected:
    class RegionInverse
public:

    typedef typename Traits::Iterator Iterator;
    typedef Iterator const_iterator;

    //@{ Region requirements.

    /// @brief Iterator over spans.
    Iterator begin() const { return self().begin(); }

    /// @brief Iterator over spans.
    Iterator end() const { return self().end(); }

    /// @brief Number of spans.
    int size() const { return self().size(); }

    /// @brief Equivalent to (size() == 0) but possibly faster.
    bool empty() const { return self().empty(); }

    /// @brief Total number of pixels.
    int getArea() const { return self().getArea(); }

    /**
     *  @brief A box that contains all spans in the region.
     *
     *  The bounding box should be a tight bounding box; there must not be any smaller bounding box
     *  that also contains all spans.
     */
    geom::Box2I getBBox() const { return self().getBBox(); }

    //@}

    /// @brief Return true if all pixels are simply connected (no holes or diagonal-only connections).
    bool isSimplyConnected() const;

    //@[
    /**
     *  @brief Iterate over all spans and pixels and call corresponding methods on the given functor.
     *
     *  Calls functor.processSpan(s) for each Span s.  If the return value is true, calls 
     *  functor.processPixel(x, y) for every x, y coordinate pair in s.  Finally, calls and returns
     *  functor.finish(n), where n is the area of the region.
     *
     *  For overloads that take additional targets, a iterator to the beginning of the span for
     *  each target is additionally passed to processSpan (that correspond to the beginning of the span),
     *  and pixel values will be passed to processPixel.
     *
     *  To pass the functor by reference or const reference, use boost::ref or boost::cref.
     */
    template <typename Functor>
    typename Functor::Result apply(Functor functor) const;

    template <typename Functor, typename Target1>
    typename Functor::Result apply(Functor functor, Target1 target1) const;

    template <typename Functor, typename Target1, typename Target2>
    typename Functor::Result apply(Functor functor, Target1 target1, Target2 target2) const;

    template <typename Functor, typename Target1, typename Target2, typename Target3>
    typename Functor::Result apply(Functor functor, Target1 target1, Target2 target2, Target3 target3) const;
    //@}

    /**
     *  @brief Return true if any pixel in this is also in other.
     *
     *  Returns false if either this or other is empty.
     */
    template <typename Other>
    bool overlaps(Region<Other> const & other) const;

    /**
     *  @brief Return true if any pixel in this is also in other.
     *
     *  Returns true if other is empty, even if this is also empty.
     */
    template <typename Other>
    bool contains(Region<Other> const & other) const;

    //@{
    /**
     *  @brief Test whether two regions are identical.
     *
     *  Equality is equivalent to this->contains(other) && other.contains(*this);
     *  two empty regions are considered equal.
     */
    template <typename Other>
    bool operator==(Region<Other> const & other) const;

    template <typename Other>
    bool operator!=(Region<Other> const & other) const;    
    //@}

    /// @brief Return the inverse of the pixel region.
    RegionInverse operator~() const;

protected:

    template <typename DerivedT> friend class Region;

    /**
     *  @brief Transform a Region by an AffineTransform.
     *
     *  Prosposed algorithm: compute the floating-point convex hull of the footprint, transform
     *  the points of the polygon, and compute the footprint that contains the transformed convex
     *  polygon (so transforming and inverse-transforming always generates a footprint that contains
     *  the original).
     * 
     *  Region is an immutable concept, so we can't make some form of this public, but we put
     *  the implementation here so subclasses can make use of it.
     */
    template <typename InputSpanList, typename OutputSpanList>
    void transform(geom::AffineTransform const & t, InputSpanList const & input, OutputSpanList &output);

    Derived const & self() const { return static_cast<Derived const &>(*this); }

};

/// Expression template for operator~.
template <typename Derived>
class Region<Derived>::RegionInverse {
public:

    explicit RegionInverse(Derived const & region) : _region(region) {}

    /// @brief Return the original pixel region reference.
    Derived const & getRegion() const { return _region; }

    /// @brief Return a copy of the original pixel region.
    Derived operator~() const { return _region; }

private:
    Derived const & _region;
};

template <typename Derived>
RegionInverse<Derived> Region<Derived>::operator~() const {
    return RegionInverse<Derived>(this->self());
}

template <typename Functor>
typename Functor::Result apply(Functor functor) const {
    int n = 0;
    for (Iterator s = begin(); s != end(); ++s) {
        if (boost::unwrap_ref(functor).processSpan(*s)) {
            for (int x = s->getX0(); x <= s->getX1(); ++x, ++n) {
                boost::unwrap_ref(functor).processPixel(x, s->getY());
            }
        } else {
            n += s->getWidth();
        }
    }
    return unwrap_ref(functor).finish(n);
}

template <typename Derived>
template <typename Functor, typename Target1>
typename Functor::Result Region<Derived>::apply(Functor functor, Target1 target1) const {
    int n = 0;
    for (Iterator s = region.begin(); s != region.end(); ++s) {
        typename IterationTraits<Target1>::Iterator t1
            = IterationTraits<Target1>::getIterator(boost::unwrap_ref(target1), n, *s);
        if (boost::unwrap_ref(functor).processSpan(*s, t1) == PROCESS_PIXELS) {
            for (int x = s->getX0(); x <= s->getX1(); ++x, ++n, ++t1) {
                boost::unwrap_ref(functor).processPixel(x, s->getY(), *t1);
            }
        } else {
            n += s->getWidth();
        }
    }
    return boost::unwrap_ref(functor).finish(n);
}

template <typename Derived>
template <typename Functor, typename Target1, typename Target2>
typename Functor::Result
Region<Derived>::apply(Functor functor, Target1 target1, Target2 target2) const {
    int n = 0;
    for (Iterator s = region.begin(); s != region.end(); ++s) {
        typename IterationTraits<Target1>::Iterator t1
            = IterationTraits<Target1>::getIterator(boost::unwrap_ref(target1), n, *s);
        typename IterationTraits<Target2>::Iterator t2
            = IterationTraits<Target2>::getIterator(boost::unwrap_ref(target2), n, *s);
        if (boost::unwrap_ref(functor).processSpan(*s, t1, t2) == PROCESS_PIXELS) {
            for (int x = s->getX0(); x <= s->getX1(); ++x, ++n, ++t1, ++t2) {
                boost::unwrap_ref(functor).processPixel(x, s->getY(), *t1, *t2);
            }
        } else {
            n += s->getWidth();
        }
    }
    return boost::unwrap_ref(functor).finish(n);
}

template <typename Derived>
template <typename Functor, typename Target1, typename Target2, typename Target3>
typename Functor::Result
Region<Derived>::apply(Functor functor, Target1 target1, Target2 target2, Target3 target3) const {
    int n = 0;
    for (Iterator s = region.begin(); s != region.end(); ++s) {
        typename IterationTraits<Target1>::Iterator t1
            = IterationTraits<Target1>::getIterator(boost::unwrap_ref(target1), n, *s);
        typename IterationTraits<Target2>::Iterator t2
            = IterationTraits<Target2>::getIterator(boost::unwrap_ref(target2), n, *s);
        typename IterationTraits<Target3>::Iterator t3
            = IterationTraits<Target3>::getIterator(boost::unwrap_ref(target3), n, *s);
        if (boost::unwrap_ref(functor).processSpan(*i, t1, t2, t3) == PROCESS_PIXELS) {
            for (int x = s->getX0(); x <= s->getX1(); ++x, ++n, ++t1, ++t2, ++t3) {
                boost::unwrap_ref(functor).processPixel(x, s->getY(), *t1, *t2, *t3);
            }
        } else {
            n += s->getWidth();
        }
    }
    return boost::unwrap_ref(functor).finish(n);
}

}}} // namespace lsst::afw::pixels

#endif // !LSST_AFW_REGION_Region_INCLUDED
