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
/**
 * \file
 * \brief Represent a set of pixels of an arbitrary shape and size
 *
 * Footprint is fundamental in astronomical image processing, as it defines what
 * is meant by a Source.
 */
#include <algorithm>
#include <list>
#include <set>
#include <cmath>
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
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

using geom::Span;

/************************************************************************************************************/
/*!
 * \brief A set of pixels in an Image
 *
 * A Footprint is a set of pixels, usually but not necessarily contiguous.
 * There are constructors to find Footprints above some threshold in an Image
 * (see FootprintSet), or to create Footprints in the shape of various
 * geometrical figures
 */
class Footprint : public lsst::daf::base::Citizen,
                  public afw::table::io::PersistableFacade<lsst::afw::detection::Footprint>,
                  public afw::table::io::Persistable
{
public:
    typedef boost::shared_ptr<Footprint> Ptr;
    typedef boost::shared_ptr<const Footprint> ConstPtr;

    /// The Footprint's Span list
    typedef std::vector<Span::Ptr> SpanList;

    /**
     * Create a Footprint
     *
     * \throws lsst::pex::exceptions::InvalidParameterException in nspan is < 0
     *
     * nspan: initial number of Span%s in this Footprint
     * region: Bounding box of MaskedImage footprint
     */
    explicit Footprint(int nspan = 0, geom::Box2I const & region=geom::Box2I());

    /**
     * Create a rectangular Footprint
     */
    explicit Footprint(afw::table::Schema const & peakSchema, int nspan=0,
                       geom::Box2I const & region=geom::Box2I());

    explicit Footprint(geom::Box2I const & bbox, geom::Box2I const & region=geom::Box2I());
    Footprint(geom::Point2I const & center, double const radius, geom::Box2I const & = geom::Box2I());
    explicit Footprint(geom::ellipses::Ellipse const & ellipse, geom::Box2I const & region=geom::Box2I());

    explicit Footprint(SpanList const & spans, geom::Box2I const & region=geom::Box2I());

    /**
     * Construct a footprint from a list of spans. Resulting Footprint is
     * not normalized.
     */
    Footprint(Footprint const & other);

    virtual ~Footprint();

    /**
     * Is this a HeavyFootprint?
     */
    virtual bool isHeavy() const { return false; }

    /** Return the Footprint's unique ID. */
    int getId() const { return _fid; }

    /** Return the Span%s contained in this Footprint. */
    SpanList& getSpans() { return _spans; }

    /** Return the Span%s contained in this Footprint. */
    const SpanList& getSpans() const { return _spans; }

    /**
     * Return the Peaks contained in this Footprint
     *
     * The peaks should be ordered by decreasing pixel intensity at the peak position (so the most negative
     * peak appears last).  Users that add new Peaks manually are responsible for maintaining this sorting.
     */
    PeakCatalog & getPeaks() { return _peaks; }
    const PeakCatalog & getPeaks() const { return _peaks; }

    /// Convenience function to add a peak (since that'd now be multiple lines without this function)
    PTR(PeakRecord) addPeak(float fx, float fy, float value);

    /// Set the Schema used by the PeakCatalog (will throw if PeakCatalog is not empty).
    void setPeakSchema(afw::table::Schema const & peakSchema) {
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
    int getNpix() const { return _area; }
    int getArea() const { return _area; }

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
     * Add a Span to a footprint, returning a reference to the new Span.
     */
    const Span& addSpan(const int y, const int x0, const int x1);
    /**
     * Add a Span to a Footprint returning a reference to the new Span
     */
    const Span& addSpan(Span const& span);
    /**
     * Add a Span to a Footprint returning a reference to the new Span
     */
    const Span& addSpan(Span const& span, int dx, int dy);

    /**
     * Add a Span to a Footprint, where the Spans MUST be added in order
     * (first in increasing y, then increasing x), and MUST NOT be
     * overlapping.  This method does NOT reset the _normalized boolean.
     *
     * This method is useful when a Footprint is being constructed in
     * such a way that Spans are guaranteed to be produced in order.
     * In that case, it is not necessary to normalize() the Footprint
     * after all the Spans have been added.  This can save some
     * computation.
     */
    const Span& addSpanInSeries(const int y, const int x0, const int x1);

    /**
     * Shift a Footprint by <tt>(dx, dy)</tt>
     *
     * dx: How much to move footprint in column direction
     * dy: How much to move in row direction
     */
    void shift(int dx, int dy);
    void shift(geom::ExtentI d) {shift(d.getX(), d.getY());}

    /// Return the Footprint's bounding box
    geom::Box2I getBBox() const { return _bbox; }

    /// Return the corners of the MaskedImage the footprints live in
    geom::Box2I const & getRegion() const { return _region; }

    /// Set the corners of the MaskedImage wherein the footprints dwell
    void setRegion(geom::Box2I const & region) { _region = region; }

    void clipTo(geom::Box2I const & bbox);

    /**
     * Clips the given *Footprint* to the region in the *Image*
     * containing non-zero values.  The clipping drops spans that are
     * totally zero, and moves endpoints to non-zero; it does not
     * split spans that have internal zeros.
     */
    template<typename PixelT>
    void clipToNonzero(lsst::afw::image::Image<PixelT> const& img);

    /**
     * Does this Footprint contain this pixel?
     */
    bool contains(geom::Point2I const& pix) const;

    /**
     * Normalise a Footprint, sorting spans and setting the BBox
     */
    void normalize();
    bool isNormalized() const {return _normalized;}

    /**
     * Set the pixels in idImage that are in Footprint by adding the
     * specified value to the Image.
     *
     * idImage: Image to contain the footprint
     * id: Add id to idImage for pixels in the Footprint
     * region: Footprint's region (default: getRegion())
     */
    template<typename PixelT>
    void insertIntoImage(typename lsst::afw::image::Image<PixelT>& idImage,
                         boost::uint64_t const id,
                         geom::Box2I const& region=geom::Box2I()
    ) const;

    /**
     * Set the pixels in idImage which are in Footprint by adding the
     * specified value to the Image.
     *
     * The list of ids found under the new Footprint are returned.
     *
     * idImage: Image to contain the footprint
     * id: Add id to idImage for pixels in the Footprint
     * overwriteId: should id replace any value already in idImage?
     * idMask: Don't overwrite ID bits in this mask
     * oldIds: if non-NULL, set the IDs that were overwritten
     * region: Footprint's region (default: getRegion())
     */
    template<typename PixelT>
    void insertIntoImage(typename lsst::afw::image::Image<PixelT>& idImage,
                         boost::uint64_t const id,
                         bool const overwriteId, long const idMask,
                         typename std::set<boost::uint64_t> *oldIds,
                         geom::Box2I const& region=geom::Box2I()
    ) const;

    /**
     * Assignment operator. Will not change the id
     */
    Footprint & operator=(Footprint & other);

    /**
     * \brief Intersect the Footprint with a Mask
     *
     * The resulting Footprint contains only pixels for which (mask & bitMask) == 0;
     * it may have disjoint pieces. Any part of the footprint that falls outside the
     * bounds of the mask will be clipped.
     *
     */
    template <typename MaskPixelT>
    void intersectMask(
        image::Mask<MaskPixelT> const & mask,
        MaskPixelT bitmask=~0x0
    );

    /**
     *  @brief Transform the footprint from one WCS to another
     *
     *  @param[in]  source   Wcs that defines the coordinate system of the input footprint.
     *  @param[in]  target   Wcs that defines that desired coordinate system of the returned footprint.
     *  @param[in]  region   Used to set the "region" box of the returned footprint; note that this is
     *                       NOT the same as the footprint's bounding box.
     *  @param[in]  doClip   If true, clip the new footprint to the region bbox before returning it.
     */
    PTR(Footprint) transform(
        image::Wcs const & source,
        image::Wcs const & target,
        geom::Box2I const & region,
        bool doClip=true
    ) const;

    /**
     *  @brief Update the Footprint in-place to be the union of itself and all others provided
     *
     *  Only spans will be modified; peaks will be left unchanged.
     *
     *  NOTE: this is for the case of contiguous sets of footprints.
     *  If the union is disjoint, throw RuntimeError Exception.
     */
    void include(std::vector<PTR(Footprint)> const & others);

    bool isPersistable() const { return true; }

protected:

    virtual std::string getPersistenceName() const;

    virtual std::string getPythonModule() const;

    virtual void write(OutputArchiveHandle & handle) const;

    //@{
    /// Persistence implementation functions made available for derived classes
    void readSpans(afw::table::BaseCatalog const & spanCat);
    void readPeaks(afw::table::BaseCatalog const & peakCat);
    //@}

    friend class FootprintFactory;

private:

    friend class FootprintMerge;

    static int id;
    mutable int _fid;                    //!< unique ID
    int _area;                           //!< number of pixels in this Footprint (not the area of the bbox)

    SpanList _spans;                     //!< the Spans contained in this Footprint
    geom::Box2I _bbox;                   //!< the Footprint's bounding box
    PeakCatalog _peaks;                     //!< the Peaks lying in this footprint
    mutable geom::Box2I _region;         //!< The corners of the MaskedImage the footprints live in
    bool _normalized;                    //!< Are the spans sorted?
};

/**
 Given a vector of Footprints, fills the output "argmin" and "dist"
 images to contain the Manhattan distance to the nearest footprint (in
 "dist") and the identity of the nearest footprint (in "argmin").

 For example, if there are two footprints at y=0 covering x=[1,2] and [7,7],

 Index     :  0 1 2 3 4 5 6 7

 Footprints:  . 0 0 . . . . 1

 Argmin    :  0 0 0 0 0 1 1 1

 Dist      :  1 0 0 1 2 2 1 0

 "argmin" gives the index of the nearest footprint, and "dist" its
 Manhattan (L_1 norm) distance.  The pixel at index 4 is closest to
 footprint 0, and its distance is 2.
 */
void nearestFootprint(std::vector<PTR(Footprint)> const& foots,
                      lsst::afw::image::Image<boost::uint16_t>::Ptr argmin,
                      lsst::afw::image::Image<boost::uint16_t>::Ptr dist);

/**
   Merges two Footprints -- appends their peaks, and unions their
   spans, returning a new Footprint.

   This const version requires that both input footprints are
   normalized (and will raise an exception if not).
 */
PTR(Footprint) mergeFootprints(Footprint const& foot1, Footprint const& foot2);

/**
   Merges two Footprints -- appends their peaks, and unions their
   spans, returning a new Footprint.
 */
PTR(Footprint) mergeFootprints(Footprint& foot1, Footprint& foot2);

/**
 * Shrink a footprint isotropically by nGrow pixels, returning a new Footprint.
 */
PTR(Footprint) shrinkFootprint(Footprint const& foot, int nGrow, bool isotropic);

/**
 * Grow a Footprint by nGrow pixels, returning a new Footprint.
 */
PTR(Footprint) growFootprint(Footprint const& foot, int nGrow, bool isotropic=true);

/**
 * \note Deprecated interface; use the Footprint const& version.
 */
PTR(Footprint) growFootprint(PTR(Footprint) const& foot, int nGrow, bool isotropic=true);

/**
 * \brief Grow a Footprint in at least one of the cardinal directions,
 * returning a new Footprint
 *
 * Note that any left/right grow is done prior to the up/down grow, so
 * any left/right grown pixels \em are subject to a further up/down
 * grow (i.e. an initial single pixel Footprint will end up as a
 * square, not a cross.
 */
PTR(Footprint) growFootprint(Footprint const& foot, int nGrow,
                             bool left, bool right, bool up, bool down);

/**
 * Return a list of BBox%s, whose union contains exactly the pixels in
 * foot, neither more nor less
 *
 * Useful in generating sets of meas::algorithms::Defects for the ISR
 */
std::vector<lsst::afw::geom::Box2I> footprintToBBoxList(Footprint const& foot);

/**
 * \brief Set all image pixels in a Footprint to a given value
 *
 * \return value
 */
template<typename ImageT>
typename ImageT::Pixel setImageFromFootprint(ImageT *image,
                                             Footprint const& footprint,
                                             typename ImageT::Pixel const value);

/**
 * \brief Set all image pixels in a set of Footprint%s to a given value
 *
 * \return value
 */
template<typename ImageT>
typename ImageT::Pixel setImageFromFootprintList(ImageT *image,
                                                 CONST_PTR(std::vector<PTR(Footprint)>) footprints,
                                                 typename ImageT::Pixel  const value);

/**
 * \brief Set all image pixels in a set of Footprint%s to a given value
 *
 * \return value
 */
template<typename ImageT>
typename ImageT::Pixel setImageFromFootprintList(ImageT *image,
                                                 std::vector<PTR(Footprint)> const& footprints,
                                                 typename ImageT::Pixel  const value);

/**
 * \brief OR bitmask into all the Mask's pixels that are in the Footprint
 *
 * \return bitmask
 */
template<typename MaskT>
MaskT setMaskFromFootprint(lsst::afw::image::Mask<MaskT> *mask,
                           Footprint const& footprint,
                           MaskT const bitmask);

/**
 * \brief (AND ~bitmask) all the Mask's pixels that are in the
 * Footprint; that is, set to zero in the Mask-intersecting-Footprint
 * all bits that are 1 in then bitmask.
 *
 * \return bitmask
 */
template<typename MaskT>
MaskT clearMaskFromFootprint(lsst::afw::image::Mask<MaskT> *mask,
                             Footprint const& footprint,
                             MaskT const bitmask);

/**
 Copies pixels from input image to output image within the Footprint's
 area.

 The input and output image must be the same type -- either Image or
 MaskedImage.
 */
template <typename ImageOrMaskedImageT>
void copyWithinFootprint(Footprint const& foot,
                         PTR(ImageOrMaskedImageT) const input,
                         PTR(ImageOrMaskedImageT) output);

/************************************************************************************************************/
/**
 * \brief OR bitmask into all the Mask's pixels which are in the set of Footprint%s
 *
 * \return bitmask
 */
template<typename MaskT>
MaskT setMaskFromFootprintList(lsst::afw::image::Mask<MaskT> *mask,
                               std::vector<PTR(Footprint)> const& footprints,
                               MaskT const bitmask);

/**
 * \brief OR bitmask into all the Mask's pixels which are in the set of Footprint%s
 *
 * \return bitmask
 */
template<typename MaskT>
MaskT setMaskFromFootprintList(lsst::afw::image::Mask<MaskT> *mask,
                               CONST_PTR(std::vector<PTR(Footprint)>) const& footprints,
                               MaskT const bitmask);

/**
 * \brief Return a Footprint that's the intersection of a Footprint with a Mask
 *
 * The resulting Footprint contains only pixels for which (mask & bitMask) != 0;
 * it may have disjoint pieces
 *
 * \note This isn't a member of Footprint as Footprint isn't templated over MaskT
 *
 * foot: The initial Footprint
 * mask: The mask to & with foot
 * bitmask: Only consider these bits
 *
 * \returns Returns the new Footprint
 */
template<typename MaskT>
PTR(Footprint) footprintAndMask(PTR(Footprint) const& foot,
                                typename image::Mask<MaskT>::Ptr const& mask,
                                MaskT const bitmask);

/************************************************************************************************************/

}}} // namespace lsst::afw::detection

#endif
