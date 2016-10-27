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
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */

#if !defined(LSST_DETECTION_FOOTPRINT_H)
#define LSST_DETECTION_FOOTPRINT_H

#include <algorithm>
#include <list>
#include <set>
#include <cmath>
#include <cstdint>
#include <memory>
#include "ndarray.h"
#include "lsst/base.h"
#include "lsst/pex/policy/Policy.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/detection/Peak.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/table/fwd.h"
#include "lsst/afw/table/io/Persistable.h"

namespace lsst { namespace afw { namespace detection {

class Footprint : public lsst::daf::base::Citizen,
                  public afw::table::io::PersistableFacade<lsst::afw::detection::Footprint>,
                  public afw::table::io::Persistable

{
public:
    // If someone has a unique_ptr, they can do a std::move to set it to the
    // shared_ptr argument
    explicit Footprint(std::shared_ptr<geom::SpanSet> inputSpans,
                       geom::Box2I const & region=geom::Box2I());
    explicit Footprint(std::shared_ptr<geom::SpanSet> inputSpans,
                       table::Schema const & peakSchema,
                       geom::Box2I const & region=geom::Box2I());

    explicit Footprint(geom::SpanSet inputSpans,
                       geom::Box2I const & region=geom::Box2I());
    explicit Footprint(geom::SpanSet inputSpans,
                       table::Schema const & peakSchema,
                       geom::Box2I const & region=geom::Box2I());

    Footprint(Footprint const & other) = default;
    Footprint(Footprint && ) = default;

    Footprint & operator=(Footprint const & other) = default;
    Footprint & operator=(Footprint && ) = default;

    virtual ~Footprint();

    /**
     * Is this a HeavyFootprint?
     */
    virtual bool isHeavy() const { return false; }

    /** Return the SpanSet */
    std::shared_ptr<geom::SpanSet> getSpans() const { return _spans;}

    /** Set the SpanSet in the footprint */
    setSpans(std::shared_ptr<geom::SpanSet> otherSpanSet);
    setSpans(geom::SpanSet otherSpanSet);

    /**
     * Return the Peaks contained in this Footprint
     *
     * The peaks should be ordered by decreasing pixel intensity at the peak position (so the most negative
     * peak appears last).  Users that add new Peaks manually are responsible for maintaining this sorting.
     */
    PeakCatalog & getPeaks() { return _peaks; }
    const PeakCatalog & getPeaks() const { return _peaks; }

    /// Convenience function to add a peak (since that'd now be multiple lines without this function)
    std::shared_ptr(PeakRecord) addPeak(float fx, float fy, float value);

    /**
     *  Sort Peaks from most positive value to most negative.
     *
     *  If the key passed is invalid (the default) PeakTable::getPeakValueKey() will be used.
     */
    void sortPeaks(table::Key<float> const & key=table::Key<float>());

    /// Set the Schema used by the PeakCatalog (will throw if PeakCatalog is not empty).
    void setPeakSchema(table::Schema const & peakSchema) {
        if (!getPeaks().empty()) {
            throw LSST_EXCEPT(
                pex::exceptions::LogicError,
                "Cannot change the PeakCatalog schema unless it is empty"
            );
        }
        // this syntax doesn't work in Python, which is why this method has to exist
        getPeaks() = PeakCatalog(peakSchema);
    }

    /**
     * Return the number of pixels in this Footprint (the real number
     * of pixels, not the area of the bbox).
     */
    size_t getArea() const;

    /**
     * Return the Footprint's centroid
     *
     * The centroid is calculated as the mean of the pixel centers
     */
    geom::Point2D getCentroid() const;

    /**
     * Return the Footprint's shape (interpreted as an ellipse)
     *
     * The shape is determined by measuring the moments of the pixel
     * centers about its centroid (cf. getCentroid)
     */
    geom::ellipses::Quadrupole getShape() const;

    /**
     * Shift a Footprint by <tt>(dx, dy)</tt>
     *
     * dx: How much to move footprint in column direction
     * dy: How much to move in row direction
     */
    void shift(int dx, int dy);
    void shift(geom::ExtentI const & d) {shift(d.getX(), d.getY());}

    /// Return the Footprint's bounding box
    geom::Box2I getBBox() const { return _bbox; }

    /// Return the corners of the MaskedImage the footprints live in
    geom::Box2I const & getRegion() const { return _region; }

    /// Set the corners of the MaskedImage wherein the footprints dwell
    void setRegion(geom::Box2I const & region) { _region = region; }


    void clipTo(geom::Box2I const & bbox);

    /**
     * Does this Footprint contain this pixel?
     */
    bool contains(geom::Point2I const& pix) const;

    /**
     *  @brief Transform the footprint from one WCS to another
     *
     *  @param[in]  source   Wcs that defines the coordinate system of the input footprint.
     *  @param[in]  target   Wcs that defines that desired coordinate system of the returned footprint.
     *  @param[in]  region   Used to set the "region" box of the returned footprint; note that this is
     *                       NOT the same as the footprint's bounding box.
     *  @param[in]  doClip   If true, clip the new footprint to the region bbox before returning it.
     */
    std::unique_ptr<Footprint> transform(
        image::Wcs const & source,
        image::Wcs const & target,
        geom::Box2I const & region,
        bool doClip=true
    ) const;

    bool isPersistable() const { return true; }

    std::unique_ptr<Footprint> intersect(std::shared_ptr<Footprint> other) const;
    template <typename T>
    std::unique_ptr<Footprint> intersect(image::Mask<T> const & other, T bitMask) const;
    std::unique_ptr<Footprint> intersectNot(std::shared_ptr<Footprint> other) const;
    std::unique_ptr<Footprint> intersectNot(image::Mask<T> const & other, T bitMask) const;
    std::unique_ptr<Footprint> union_(std::shared_ptr<Footprint> other) const;
    std::unique_ptr<Footprint> union_(image::Mask<T> const & other, T bitMask) const;

    void dilate(int r, geom::Stencil s = geom::Stencil::CIRCLE);
    void dilate(geom::SpanSet const & other);

    void erode(int r, geom::Stencil s = geom::Stencil::CIRCLE);
    void erode(SpanSet const & other);

 protected:

    virtual std::string getPersistenceName() const;

    virtual std::string getPythonModule() const;

    virtual void write(OutputArchiveHandle & handle) const;

    //@{
    /// Persistence implementation functions made available for derived classes
    void readSpans(table::BaseCatalog const & spanCat);
    void readPeaks(table::BaseCatalog const & peakCat);
    //@}

    friend class FootprintFactory;

 private:

    friend class FootprintMerge; // Maybe make it not a friend

    std::shared_ptr<SpanSet> _spans;    //!< The SpanSet representing area on image
    PeakCatalog _peaks;                 //!< The peaks lying in this footprint
    geom::Box2I _region;     //!< The corners of the MaskedImage the footprints live in
};

}}} // Close namespace lsst::afw::detection
