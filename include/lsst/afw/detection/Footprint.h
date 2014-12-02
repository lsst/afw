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

namespace lsst {
namespace afw { 
namespace detection {

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
    typedef std::vector<Peak::Ptr> PeakList;

    explicit Footprint(int nspan = 0, geom::Box2I const & region=geom::Box2I());
    explicit Footprint(geom::Box2I const & bbox, geom::Box2I const & region=geom::Box2I());
    Footprint(geom::Point2I const & center, double const radius, geom::Box2I const & = geom::Box2I());
    explicit Footprint(geom::ellipses::Ellipse const & ellipse, geom::Box2I const & region=geom::Box2I());

    explicit Footprint(SpanList const & spans, geom::Box2I const & region=geom::Box2I());
    Footprint(Footprint const & other);
    virtual ~Footprint();

    virtual bool isHeavy() const { return false; }

    int getId() const { return _fid; }   //!< Return the Footprint's unique ID
    SpanList& getSpans() { return _spans; } //!< return the Span%s contained in this Footprint
    const SpanList& getSpans() const { return _spans; } //!< return the Span%s contained in this Footprint
    /**
     * Return the Peak%s contained in this Footprint
     *
     * The peaks are ordered by decreasing pixel intensity at the peak position (so the most negative
     * peak appears last).
     */
    PeakList & getPeaks() { return _peaks; }
    const PeakList & getPeaks() const { return _peaks; } //!< Return the Peak%s contained in this Footprint
    int getNpix() const { return _area; }     //!< Return the number of pixels in this Footprint (the real number of pixels, not the area of the bbox)
    int getArea() const { return _area; }
    geom::Point2D getCentroid() const;
    geom::ellipses::Quadrupole getShape() const;

    const Span& addSpan(const int y, const int x0, const int x1);
    const Span& addSpan(Span const& span);
    const Span& addSpan(Span const& span, int dx, int dy);

    const Span& addSpanInSeries(const int y, const int x0, const int x1);

    void shift(int dx, int dy);
    void shift(geom::ExtentI d) {shift(d.getX(), d.getY());}

    /// Return the Footprint's bounding box
    geom::Box2I getBBox() const { return _bbox; }
    /// Return the Footprint's bounding box
    geom::Box2I & getBBox() { return _bbox; }
    /// Return the corners of the MaskedImage the footprints live in
    geom::Box2I const & getRegion() const { return _region; }

    /// Set the corners of the MaskedImage wherein the footprints dwell
    void setRegion(geom::Box2I const & region) { _region = region; }

    void clipTo(geom::Box2I const & bbox);

    template<typename PixelT>
    void clipToNonzero(lsst::afw::image::Image<PixelT> const& img);

    bool contains(geom::Point2I const& pix) const;

    void normalize();
    bool isNormalized() const {return _normalized;}

    template<typename PixelT>
    void insertIntoImage(typename lsst::afw::image::Image<PixelT>& idImage,
                         boost::uint64_t const id,
                         geom::Box2I const& region=geom::Box2I()
    ) const;
    template<typename PixelT>
    void insertIntoImage(typename lsst::afw::image::Image<PixelT>& idImage,
                         boost::uint64_t const id,
                         bool const overwriteId, long const idMask,
                         typename std::set<boost::uint64_t> *oldIds,
                         geom::Box2I const& region=geom::Box2I()
    ) const;

    Footprint & operator=(Footprint & other);

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
     *  @brief Update the Footprint in-place to be the union of itself and all its children
     *
     *  Only spans will be modified; peaks will be left unchanged.
     *
     *  If the union of all children with this is disjoint, throw RuntimeErrorException.
     */
    void include(std::vector<PTR(Footprint)> const & children);

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

    static int id;
    mutable int _fid;                    //!< unique ID
    int _area;                           //!< number of pixels in this Footprint (not the area of the bbox)

    SpanList _spans;                     //!< the Spans contained in this Footprint
    geom::Box2I _bbox;                   //!< the Footprint's bounding box
    PeakList _peaks;                     //!< the Peaks lying in this footprint
    mutable geom::Box2I _region;         //!< The corners of the MaskedImage the footprints live in
    bool _normalized;                    //!< Are the spans sorted?
};

void nearestFootprint(std::vector<Footprint::Ptr> const& foots,
                      lsst::afw::image::Image<boost::uint16_t>::Ptr argmin,
                      lsst::afw::image::Image<boost::uint16_t>::Ptr dist);

Footprint::Ptr mergeFootprints(Footprint const& foot1, Footprint const& foot2);
Footprint::Ptr mergeFootprints(Footprint& foot1, Footprint& foot2);

Footprint::Ptr growFootprint(Footprint const& foot, int ngrow, bool isotropic=true);
Footprint::Ptr growFootprint(Footprint::Ptr const& foot, int ngrow, bool isotropic=true);
Footprint::Ptr growFootprint(Footprint const& foot, int ngrow,
                             bool left, bool right, bool up, bool down);

std::vector<lsst::afw::geom::Box2I> footprintToBBoxList(Footprint const& foot);

template<typename ImageT>
typename ImageT::Pixel setImageFromFootprint(ImageT *image,
                                             Footprint const& footprint,
                                             typename ImageT::Pixel const value);
template<typename ImageT>
typename ImageT::Pixel setImageFromFootprintList(ImageT *image,
                                                 CONST_PTR(std::vector<Footprint::Ptr>) footprints,
                                                 typename ImageT::Pixel  const value);
template<typename ImageT>
typename ImageT::Pixel setImageFromFootprintList(ImageT *image,
                                                 std::vector<Footprint::Ptr> const& footprints,
                                                 typename ImageT::Pixel  const value);
template<typename MaskT>
MaskT setMaskFromFootprint(lsst::afw::image::Mask<MaskT> *mask,
                           Footprint const& footprint,
                           MaskT const bitmask);
template<typename MaskT>
MaskT clearMaskFromFootprint(lsst::afw::image::Mask<MaskT> *mask,
                             Footprint const& footprint,
                             MaskT const bitmask);

/// Copy pixels defined by a footprint between images
///
/// Respects image xy0, so the footprints should be defined in the
/// "PARENT" frame.
///
/// Pixels that are in the footprint but do not overlap with both
/// images are not copied.
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
                               std::vector<Footprint::Ptr> const& footprints,
                               MaskT const bitmask);
template<typename MaskT>
MaskT setMaskFromFootprintList(lsst::afw::image::Mask<MaskT> *mask,
                               CONST_PTR(std::vector<Footprint::Ptr>) const & footprints,
                               MaskT const bitmask);
template<typename MaskT>
Footprint::Ptr footprintAndMask(Footprint::Ptr const&  foot,
                                typename image::Mask<MaskT>::Ptr const&  mask,
                                MaskT const bitmask);

/************************************************************************************************************/

}}}

#endif
