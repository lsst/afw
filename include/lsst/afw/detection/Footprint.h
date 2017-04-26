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
#include "lsst/afw/geom/Span.h"
#include "lsst/afw/geom/SpanSet.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/geom/LinearTransform.h"
#include "lsst/afw/table/fwd.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/io/InputArchive.h"

namespace lsst { namespace afw { namespace detection {

/** @class lsst::afw::detection::Footprint
 * @brief Class to describe the properties of a detected object from and image
 *
 * A Footprint is designed to be constructed with information about a detected
 * object in an image. Internally a Footprint holds a SpanSet which is used to
 * describe the location of the object in the image (the x, y pixel locations
 * which are considered part of the object). In addition a Footprint contains
 * a PeakCatalog which is used to store the location and intensity of peaks in
 * the detection.
 */
class Footprint : public lsst::daf::base::Citizen,
                  public afw::table::io::PersistableFacade<lsst::afw::detection::Footprint>,
                  public afw::table::io::Persistable
{
public:
    /*
     * As a note there is no constructor which accepts a unique_ptr to a SpanSet.
     * If someone did have a unique_ptr to a SpanSet it is possible to wrap the
     * unique_ptr in a std::move when calling the constructor of a Footprint and
     * the compiler will implicitly turn the unique_ptr into a shared_ptr and use
     * that to construct the Footprint.
     */

    explicit Footprint(std::shared_ptr<geom::SpanSet> inputSpans,
                       geom::Box2I const & region=geom::Box2I());

    explicit Footprint(std::shared_ptr<geom::SpanSet> inputSpans,
                       afw::table::Schema const & peakSchema,
                       geom::Box2I const & region=geom::Box2I());

    explicit Footprint(): lsst::daf::base::Citizen(typeid(this)),
                          _spans(std::make_shared<geom::SpanSet>()),
                          _peaks(PeakTable::makeMinimalSchema()),
                          _region(geom::Box2I()) {}


    Footprint(Footprint const & other) = default;
    Footprint(Footprint &&) = default;

    Footprint & operator=(Footprint const & other) = default;
    Footprint & operator=(Footprint &&) = default;

    virtual ~Footprint() {}

    /** @brief Indicates if this object is a HeavyFootprint
     */
    virtual bool isHeavy() const { return false; }

    /** @brief Return a shared pointer to the SpanSet
      */
    std::shared_ptr<geom::SpanSet> getSpans() const { return _spans;}

    /** @brief Sets the shared pointer to the SpanSet in the Footprint
     *
     * @param otherSpanSet - Shared pointer to a SpanSet
     */
    void setSpans(std::shared_ptr<geom::SpanSet> otherSpanSet);

    /**
     * @brief Return the Peaks contained in this Footprint
     *
     * The peaks should be ordered by decreasing pixel intensity at the peak position (so the most negative
     * peak appears last).  Users that add new Peaks manually are responsible for maintaining this sorting.
     */
    PeakCatalog & getPeaks() { return _peaks; }
    const PeakCatalog & getPeaks() const { return _peaks; }

    /** @brief Convenience function to add a peak
     *
     * @param fx - Float containing the x position of a peak
     * @param fy - Float containing the y position of a peak
     * @param value - The intensity value of the peak
     */
    std::shared_ptr<PeakRecord> addPeak(float fx, float fy, float value);

    /**
     *  @brief Sort Peaks from most positive value to most negative.
     *
     *  If the key passed is invalid (the default) PeakTable::getPeakValueKey() will be used.
     *
     * @param key - A key corresponding to the field in the Scheam the PeakCatalog is to be
     *              sorted by.
     */
    void sortPeaks(afw::table::Key<float> const & key=afw::table::Key<float>());

    /** @brief Set the Schema used by the PeakCatalog (will throw if PeakCatalog is not empty).
     *
     * This function will throw a pex::eceptions::LogicError if the PeakCatalog is not empty
     *
     * @param peakSchema - The schema to use in the PeakCatalog
     */
    void setPeakSchema(afw::table::Schema const & peakSchema);

    /** @brief Set the peakCatalog to a copy of the supplied catalog
     *
     * PeakCatalog will be copied into the Footprint, but a PeakCatalog is a shallow
     * copy, so records will not be duplicated. This function will throw an error if
     * the PeakCatalog of *this is not empty.
     *
     * @param otherPeaks The PeakCatalog to copy
     */
    void setPeakCatalog(PeakCatalog const & otherPeaks);

    /**
     * @brief Return the number of pixels in this Footprint
     *
     * This function returns the real number of pixels, not the area of the bbox.
     */
    std::size_t getArea() const { return _spans->getArea(); }

    /**
     * @brief Return the Footprint's centroid
     *
     * The centroid is calculated as the mean of the pixel centers
     */
    geom::Point2D getCentroid() const { return _spans->computeCentroid(); }

    /**
     * @brief Return the Footprint's shape (interpreted as an ellipse)
     *
     * The shape is determined by measuring the moments of the pixel
     * centers about its centroid (cf. getCentroid)
     */
    geom::ellipses::Quadrupole getShape() const { return _spans->computeShape(); }

    /**
     * @brief Shift a Footprint by <tt>(dx, dy)</tt>
     *
     * @param dx - How much to move Footprint in column direction
     * @param dy - How much to move in row direction
     */
    void shift(int dx, int dy);

    /**
    * @brief Shift a Footprint by a given extent
    *
    * @param d - ExtentI object which gives the dimensions the Footprint should be shifted
    */
    void shift(geom::ExtentI const & d) { shift(d.getX(), d.getY()); }

    /**
     * @brief Return the Footprint's bounding box
     */
    geom::Box2I getBBox() const { return _spans->getBBox(); }

    /**
     * @brief Return the corners of the MaskedImage the footprints live in
     */
    geom::Box2I getRegion() const { return _region; }

    /**
     * @brief Set the corners of the MaskedImage wherein the footprints dwell
     *
     * @param region - A box describing the corners of the Image the Footprint derives from
     */
    void setRegion(geom::Box2I const & region) { _region = region; }


    /**
     * @brief Clip the Footprint such that all values lie inside the supplied Bounding Box
     *
     * @param bbox - Integer box object that defines the boundarys the footprint should be
     *               clipped to.
     */
    void clipTo(geom::Box2I const & bbox);

    /**
     * @brief Tests if a pixel postion falls inside the Footprint
     *
     * @param pix - Integer point object defining the position of a pixel to test
     */
    bool contains(geom::Point2I const & pix) const;

    /**
     *  @brief Transform the footprint from one WCS to another
     *
     *  @param source - Wcs that defines the coordinate system of the input footprint.
     *  @param target - Wcs that defines that desired coordinate system of the returned footprint.
     *  @param region - Used to set the "region" box of the returned footprint; note that this is
     *                  NOT the same as the footprint's bounding box.
     *  @param doClip - If true, clip the new footprint to the region bbox before returning it.
     */
    std::shared_ptr<Footprint> transform(
        std::shared_ptr<image::Wcs> source,
        std::shared_ptr<image::Wcs> target,
        geom::Box2I const & region,
        bool doClip=true
    ) const;

    /** Return a new Footprint who's pixels are the product of applying the specified transformation
     *
     * @param t A linear transform object which will be used to map the pixels
     * @param region Used to set the "region" box of the returned footprint; note that this is
     *               NOT the same as the footprint's bounding box.
     * @param doClip If true, clip the new footprint to the region bbox before returning it.
     */
    std::shared_ptr<Footprint> transform(geom::LinearTransform const & t,
                                         geom::Box2I const & region, bool doCLip=true) const;

    /** Return a new Footprint who's pixels are the product of applying the specified transformation
     *
     * @param t An affine transform object which will be used to map the pixels
     * @param region Used to set the "region" box of the returned footprint; note that this is
     *               NOT the same as the footprint's bounding box.
     * @param doClip If true, clip the new footprint to the region bbox before returning it.
     */
    std::shared_ptr<Footprint> transform(geom::AffineTransform const & t,
                                         geom::Box2I const & region, bool doClip=true) const;

    /** Return a new Footprint who's pixels are the product of applying the specified transformation
     *
     * @param t A XY transform object which will be used to map the pixels
     * @param region Used to set the "region" box of the returned footprint; note that this is
     *               NOT the same as the footprint's bounding box.
     * @param doClip If true, clip the new footprint to the region bbox before returning it.
     */
    std::shared_ptr<Footprint> transform(geom::XYTransform const & t,
                                         geom::Box2I const & region, bool doClip=true) const;

    /**
     * @brief Reports if this object is persistable
     */
    bool isPersistable() const { return true; }

    /**
     * @brief Dilate the Footprint with a defined kernel
     *
     * This function enlarges the SpanSet which defines the area of the Footprint by
     * an amount governed by in input kernel
     *
     * @param r - The radius of the stencil object used to create a dilation kernel
     * @param s - The stencil object used to create the dilation kernel
     */
    void dilate(int r, geom::Stencil s = geom::Stencil::CIRCLE);

    /**
     * @brief Dilate the Footprint with a defined kernel
     *
     * This function enlarges the SpanSet which defines the area of the Footprint by
     * an amount governed by the input kernel
     *
     * @param other - SpanSet to use as the kernel in dilation
     */
    void dilate(geom::SpanSet const & other);

    /**
     * @brief Erode the Footprint with a defined kernel
     *
     * This function reduces the size of the SpanSet which defines the area of the Footprint
     * by an amount governed by the input kernel
     *
     * @param r - The radius of the stencil object used to create a erosion kernel
     * @param s - The stencil object used to create the erosion kernel
     */
    void erode(int r, geom::Stencil s = geom::Stencil::CIRCLE);

    /**
     * @brief Erode the Footprint with a defined kernel
     *
     * This function reduces the size of the SpanSet which defines the area of the Footprint
     * by an amount governed by the input kernel
     *
     * @param other - SpanSet to use as the kernel in erosion
     */
    void erode(geom::SpanSet const & other);

    /**
     * @brief Remove peaks from the PeakCatlog that fall ouside the area of the Footprint
     */
    void removeOrphanPeaks();

    /**
     * @brief Reports if the Footprint is simply connected or has multiple components
     */
    bool isContiguous() const { return getSpans()->isContiguous(); };

    /**
    * @brief Split a multi-component Footprint into a vector of contiguous Footprints
    *
    * Split a multi-component Footprint such that each Footprint in the output vector
    * is contiguous and contains only peaks that can be found within the bounds of the
    * Footprint
    */
    std::vector<std::shared_ptr<Footprint>> split() const;

    /**
    * @brief equality operator
    *
    * @param other - The Footprint for which equality will be computed
    */
    bool operator==(Footprint const & other) const;

 protected:

    /*
     * Return the name correspoinging ot the persistence type
     */
    std::string getPersistenceName() const override;

    /*
     * Return the python module the object will live in
     */
    inline std::string getPythonModule() const override { return "lsst.afw.detection"; }

    /*
     * Write an instance of a Footprint to an output Archive
     */
    void write(OutputArchiveHandle & handle) const override;

    friend class FootprintFactory;

    /*
     * Static method used to unpersist the SpanSet member of the Footprint class
     */
    static std::unique_ptr<Footprint> readSpanSet(afw::table::BaseCatalog const &,
                                                  afw::table::io::InputArchive const &);
    /*
     * Static method used to unpersit the PeakCatalog member of the Footprint class
     */
    static void readPeaks(afw::table::BaseCatalog const &, Footprint &);

 private:

    friend class FootprintMerge;

    std::shared_ptr<geom::SpanSet> _spans;    //!< The SpanSet representing area on image
    PeakCatalog _peaks;                 //!< The peaks lying in this footprint
    geom::Box2I _region;     //!< The corners of the MaskedImage the footprints live in
};

/**
 * Merges two Footprints -- appends their peaks, and unions their
 * spans, returning a new Footprint. Region is not preserved, and is set to an empty
 * Box2I object.
 */
std::shared_ptr<Footprint> mergeFootprints(Footprint const& footprint1, Footprint const& footprint2);

/**
 * @brief Return a list of BBox%s, whose union contains exactly the pixels in
 * the footprint, neither more nor less
 *
 * Useful in generating sets of meas::algorithms::Defects for the ISR
 *
 * @param footprint Footprint to turn into bounding box list
 */
std::vector<lsst::afw::geom::Box2I> footprintToBBoxList(Footprint const& footprint);

}}} // Close namespace lsst::afw::detection

#endif // LSST_DETECTION_FOOTPRINT_H
