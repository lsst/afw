//  -*- lsst-c++ -*-
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
 
#if !defined(LSST_DETECTION_FOOTPRINT_SET_H)
#define LSST_DETECTION_FOOTPRINT_SET_H

namespace lsst { namespace afw { namespace detection {

/**
 *  A set of non-overlapping Footprints.
 *
 *  FootprintSet is a special container that maintains the property that all of
 *  its constituent Footprints are distinct (non-overlapping, non-touching),
 *  automatically merging or splitting Footprints as necessary to maintain this
 *  condition (and the requirement that Footprints be contiguous individually).
 *
 *  FootprintSets may not contain HeavyFootprints - there are some FootprintSet
 *  operations (e.g. dilate) that we can't apply to HeavyFootprints, and in
 *  general by the time we build HeavyFootprints in the pipeline, we've switched
 *  to SourceCatalog for storing them.
 */
class FootprintSet {
public:

    //@{
    /**
     *  STL container interface.
     *
     *  Instead of providing accessors to its internal container of PTR(Footprint)s,
     *  FootprintSet now just provides iterators and a bit more of the STL container
     *  interface.
     *
     *  Exactly what kind of container we use under the hood is unspecified, but
     *  the iterators will dereference to PTR(Footprint), not Footprint, as before.
     *
     *  It has no non-const iterator, because we don't want to have to deal with
     *  maintaining invariants (namely, no overlapping Footprints) while exposing
     *  references to innards.
     *
     *  Any mutating operation may invalidate iterators or Footprint references.
     *
     *  see also insert(), further below.
     */
    typedef <unspecified> const_iterator;
    typedef <unspecified> size_type;
    typedef PTR(Footprint) value_type;
    typedef value_type const & const_reference;
    const_iterator begin() const;   // iterators available only via __iter__ in Python
    const_iterator end() const;
    const_iterator cbegin() const;
    const_iterator cend() const;
    const_reference front() const;
    size_type size() const;    // also available as __len__ in Python
    bool empty() const;        // also available as !__nonzero__ in Python
    void swap(FootprintSet & rhs);
    //@}

    /// Construct an empty FootprintSet.
    FootprintSet();

    /**
     *  Construct a FootprintSet from an iterator over PTR(Footprints)
     *
     *  In Python, this is will accept any Python iterator that yields spans.
     */
    template <typename Iter>
    static FootprintSet fromFootprints(Iter first, Iter last);

#ifndef SWIG
    /// Construct a FootprintSet from an explicit list of Footprints
    static FootprintSet fromFootprints(std::initializer_list<PTR(Footprint)> footprints);
#endif

    // FootprintSet is both copyable and moveable, with no cleverness involved.
    // (it's just a thin wrapper around an STL container).
    FootprintSet(Footprint const &) = default;
    FootprintSet(Footprint &&) = default;
    FootprintSet & operator=(Footprint const &) = default;
    FootprintSet & operator=(Footprint &&) = default;

    /**
     *  Create contiguous Footprints with peaks from intermediate detection outputs.
     *
     *  This method takes a SpanRegion that identifies above-threshold pixels and splits
     *  it into contiguous Footprints.  It then finds peaks within each Footprint (which
     *  is why we need the image).
     */
    template <typename ImagePixelT>
    static FootprintSet makeFootprints(
        SpanRegion const & spans,
        image::Image<ImagePixelT> const & image,
        bool findPeaks=true
    );

    /// Create contiguous Footprints (without peaks) from a possibly noncontiguous SpanRegion.
    template <typename ImagePixelT>
    static FootprintSet makeFootprints(SpanRegion const & spans);

    /// Return a SpanRegion that contains all Spans in all Footprints in the FootprintSet.
    SpanRegion asSpanRegion() const;

    /**
     *  Add a new Footprint to the set.
     *
     *  This method isn't guaranteed to increase the number of Footprints, and in fact it
     *  might decrease it; if the new Footprint overlaps one or more already in the FootprintSet,
     *  they'll be merged.
     */
    void insert(PTR(Footprint) const & span); // also available as

    /// Move all Footprints by the given offset.
    FootprintSet shiftedBy(geom::Extent2I const & offset) const;     // return shifted copy
    FootprintSet & shiftBy(geom::Extent2I const & offset);           // in-place

    /// Clip all Footprints to the given box.
    FootprintSet clippedTo(geom::Box2I const & box) const;           // return clipped copy
    FootprintSet & clipTo(geom::Box2I const & box);                  // in-place

    //@{
    /**
     *  Mathematical morphology operations.
     *
     *  These simply delegate to the implementations for SpanRegion, but they have the
     *  potential to merge (for dilate) or split (for erode) Footprints.  The erode
     *  methods also remove peaks that, after the operation, no longer lie within their Footprint.
     */
    FootprintSet dilatedBy(int r, Stencil=Stencil.CIRCLE) const; // return dilated copy
    FootprintSet erodedBy(int r, Stencil=Stencil.CIRCLE) const;  // return eroded copy
    FootprintSet & dilateBy(int r, Stencil=Stencil.CIRCLE);      // in-place
    FootprintSet & erodeBy(int r, Stencil=Stencil.CIRCLE);       // in-place
    //@}

};

}}}

#endif
