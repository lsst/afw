// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#ifndef LSST_AFW_GEOM_SpanPixelIterator_h_INCLUDED
#define LSST_AFW_GEOM_SpanPixelIterator_h_INCLUDED

#include "boost/iterator/iterator_facade.hpp"

#include "lsst/afw/geom/Point.h"

namespace lsst { namespace afw { namespace geom {

/**
 *  @brief An iterator that yields Point2I and increases in the x direction.
 *
 *  This is used to iterate over the pixels in a Span, and by extension to iterate over
 *  regions like boxes and ellipses.
 */
class SpanPixelIterator :
        public boost::iterator_facade<SpanPixelIterator,Point2I const,boost::random_access_traversal_tag>
{
public:

    explicit SpanPixelIterator(Point2I const & p = Point2I()) : _p(p) {}

private:

    friend class boost::iterator_core_access;

    Point2I const & dereference() const { return _p; }

    void increment() { ++_p.getX(); }

    void decrement() { --_p.getX(); }

    void advance(int n) { _p.getX() += n; }

    bool equal(SpanPixelIterator const & other) const {
        return _p.getX() == other._p.getX() && _p.getY() == other._p.getY();
    }

    int distance_to(SpanPixelIterator const & other) const {
        assert(other._p.getY() == _p.getY());
        return other._p.getX() - _p.getX();
    }

    Point2I _p;
};

}}} // namespace lsst::afw::geom

#endif // !LSST_AFW_GEOM_SpanPixelIterator_h_INCLUDED
