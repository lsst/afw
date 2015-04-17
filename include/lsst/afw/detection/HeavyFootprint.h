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
 
#if !defined(LSST_DETECTION_HEAVY_FOOTPRINT_H)
#define LSST_DETECTION_HEAVY_FOOTPRINT_H

#include "lsst/afw/detection/Footprint.h"

namespace lsst { namespace afw { namespace detection {

/**
 *  Post-deblend representation of a source.
 *
 *  A HeavyFootprint contains a regular Footprint with a flattened 1-d representation of its pixel values.
 *
 *  Like Footprint, the new HeavyFootprint is immutable and always held by shared_ptr.
 *
 *  There are two additional major changes from the old HeavyFootprint:
 *   - Instead of holding (image, mask, variance) pixels, we now just hold image pixels.  As far as I can
 *     tell, we never used the mask or variance pixels in the old HeavyFootprints, so they were just taking
 *     up space unnecessarily (and quite a bit of it, actually - this could shrink our src catalogs by
 *     as much as 30%%).
 *   - Instead of templates, we just typedef the Pixel type to single-precision float.  We never used
 *     any other pixel type, and in fact never wrote working persistence code for any other pixel type.
 */
class HeavyFootprint : public Footprint {
public:

    typedef float Pixel;

#if SWIG
    /// Create a new HeavyFootprint from an existing data array.
    static PTR(HeavyFootprint) make(
        Footprint const & footprint,
        ndarray::Array<Pixel const,1,1> && pixels
    );
#endif

    /// Create a new HeavyFootprint from a copy of an existing data array.
    static PTR(HeavyFootprint) make(
        Footprint const & footprint,
        ndarray::Array<Pixel const,1,1> const & pixels
    );

    /// Create a new HeavyFootprint by extracting pixels from an image.
    static HeavyFootprint fromImage(Footprint const & footprint, afw::image::Image<Pixel> const & image);

    // HeavyFootprint is not copyable or moveable: it's immutable, and always held by shared_ptr,
    // so there's never any need to copy or move it.
    HeavyFootprint(HeavyFootprint const &) = delete;
    HeavyFootprint(HeavyFootprint &&) = delete;
    HeavyFootprint & operator=(HeavyFootprint const &) = delete;
    HeavyFootprint & operator=(HeavyFootprint &&) = delete;

    /**
     * Is this a HeavyFootprint (yes!)
     */
    virtual bool isHeavy() const { return true; }

    /**
     *  Return an equivalent Footprint that is not Heavy.
     */
    virtual PTR(Footprint) withoutPixels() const;

    /**
     *  More readable shortcut for for HeavyFootprint::make(*this, pixels).
     */
    PTR(HeavyFootprint) withNewPixels(ndarray::Array<Pixel const,1,1> const & pixels) const;
#ifndef SWIG
    PTR(HeavyFootprint) withNewPixels(ndarray::Array<Pixel const,1,1> && pixels) const;
#endif

    /**
     * Replace all the pixels in the image with the values in the HeavyFootprint.
     */
    void insert(afw::image::Image<Pixel> & image) const;

    /// Return the data array that holds the HeavyFootprints pixels, flattened.
    ndarray::Array<Pixel const,1,1> getPixels() const;

    /**
     *  Create a new Footprint by shifting this one's peaks and spans by the given amount.
     *
     *  If offset is zero, may return this.
     */
    PTR(HeavyFootprint) shiftedBy(geom::Extent2I const & offset) const {
        return static_pointer_cast<HeavyFootprint>(_shiftedBy(offset));
    }

    /**
     *  Create a new Footprint by shifting this one's peaks and spans by the given amount.
     *
     *  If box already contains the HeavyFootprint, may return this.
     */
    PTR(HeavyFootprint) clippedTo(geom::Box2I const & box) const {
        return static_pointer_cast<HeavyFootprint>(_clippedTo(box));
    }

    /**
     *  Sum two HeavyFootprints.
     *
     *  The returned HeavyFootprint will have a SpanRegion equal to the union
     *  of the SpanRegions of the inputs, with pixel values summed where they
     *  overlap.  Peak lists are concatenated.
     *
     *  @throw InvalidParameterError if the SpanRegion union is noncontiguous.
     */
    PTR(HeavyFootprint) operator+(HeavyFootprint const & rhs);

protected:

    virtual PTR(Footprint) _shiftedBy(geom::Extent2I const & offset) const;
    virtual PTR(Footprint) _clippedTo(geom::Box2I const & box) const;

};

}}}

#endif
