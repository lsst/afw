/* 
 * LSST Data Management System
 * Copyright 2008-2015 LSST Corporation.
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
 
#if !defined(LSST_DETECTION_FOOTPRINT_H)
#define LSST_DETECTION_FOOTPRINT_H

#include "lsst/afw/geom/SpanRegion.h"
#include "lsst/afw/detection/PeakRecord.h"  // PeakRecord is from HSC, see DM-1943, RFC-30

namespace lsst { namespace afw { namespace detection {

using geom::Span;

/**
 *  Detection region and peaks for a source.
 *
 *  Footprint is a combination of a SpanRegion that represents a region of above-threshold pixels
 *  and a PeakCatalog that represents the peaks within that region.
 *
 *  The biggest difference from the old Footprint class is that many operations have been moved
 *  to the new SpanRegion class, as they have nothing to do with peaks.  There are also several
 *  other differences:
 *   - The new Footprint class is immutable.  That makes it much safer to pass
 *     around and hold by shared_ptr.  (It's actually only logically immutable in this design,
 *     as its peaks can still be modified - I intend to address that in a future RFC).
 *   - Footprints *must* now be held by shared_ptr, which is enforced by the fact that they
 *     have no public constructors (only static methods that return shared_ptrs).  This allows
 *     us to use enable_shared_from_this to implement certain methods more efficiently.
 *   - Footprints are not copyable or moveable (they don't need to be, since they're always held
 *     by shared_ptr and are immutable).  That eliminates type-slicing problems with HeavyFootprint,
 *     as well as just enforcing consistent usage.
 *   - Footprint no longer holds a "region" bounding box to its parent image.  I don't think
 *     any Footprint operations actually use it, and it's not something a Footprint really
 *     needs to store itself.
 *   - Footprint no longer holds an ID (other than its Citizen ID); this has been replaced by
 *     the IDs on SourceRecord.
 */
class Footprint : private boost::enable_shared_from_this<Footprint> {
public:

    /**
     *  Construct from a SpanRegion and a PeakCatalog.
     *
     *  We have a static member function instead of a constructor so we can ensure that
     *  Footprints are always held by shared_ptr (a requirement of enable_shared_from_this).
     */
    static PTR(Footprint) make(geom::SpanRegion const & spans, PeakCatalog const & peaks);

    // Footprint is not copyable or moveable: it's immutable, and always held by shared_ptr,
    // so there's never any need to copy or move it.
    Footprint(Footprint const &) = delete;
    Footprint(Footprint &&) = delete;
    Footprint & operator=(Footprint const &) = delete;
    Footprint & operator=(Footprint &&) = delete;

    /// Return true if this is a HeavyFootprint (will be overridden by HeavyFootprint).
    virtual bool isHeavy() const { return false; }

    /**
     *  Return an equivalent Footprint that is not Heavy (returns this, if !isHeavy())
     */
    virtual PTR(Footprint) withoutPixels() const { return shared_from_this(); }

    /// Return the SpanRegion that represents the footprint area (also available as .spans in Python)
    geom::SpanRegion const & getSpans() const;

    /// Return a catalog of peaks within the footprint area (also available as .peaks in Python)
    PeakCatalog const & getPeaks() const;

    /// Convenience method for getSpans().getArea()
    std::size_t getArea() const { return getSpans().getArea(); }

    /// Convenience method for getSpans().getBBox()
    geom::Box2I getBBox() const { return getSpans().getBBox(); }

    /**
     *  Create a new Footprint by shifting this one's peaks and spans by the given amount.
     *
     *  If this is a HeavyFootprint, a HeavyFootprint will be returned.
     *
     *  If offset is zero, may return this.
     */
    PTR(Footprint) shiftedBy(geom::Extent2I const & offset) const { return _shiftedBy(offset); }

    /**
     *  Create a new Footprint by shifting this one's peaks and spans by the given amount.
     *
     *  If this is a HeavyFootprint, a HeavyFootprint will be returned.
     *
     *  If box already contains the Footprint, may return this.
     */
    PTR(Footprint) clippedTo(geom::Box2I const & box) const { return _clippedTo(box); }

protected:

    virtual PTR(Footprint) _shiftedBy(geom::Extent2I const & offset) const;
    virtual PTR(Footprint) _clippedTo(geom::Box2I const & box) const;

};

}}} // namespace lsst::afw::region
