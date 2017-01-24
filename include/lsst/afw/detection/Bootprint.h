
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
#if !defined(LSST_DETECTION_BOOTPRINT_H)
#define LSST_DETECTION_BOOTPRINT_H

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
#include "lsst/afw/table/fwd.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/io/InputArchive.h"

namespace lsst { namespace afw { namespace detection {

/** @class lsst::afw::detection::Bootprint
 * @brief Class to describe the properties of a detected object from and image
 *
 * A Bootprint is designed to be constructed with information about a detected
 * object in an image. Internally a Bootprint holds a SpanSet which is used to
 * describe the location of the object in the image (the x, y pixel locations
 * which are considered part of the object). In addition a Bootprint contains
 * a PeakCatalog which is used to store the location and intensity of peaks in
 * the detection.
 */
class Bootprint : public lsst::daf::base::Citizen,
                  public afw::table::io::PersistableFacade<lsst::afw::detection::Bootprint>,
                  public afw::table::io::Persistable
{
public:
    /*
     * As a note there is no constructor which accepts a unique_ptr to a SpanSet.
     * If someone did have a unique_ptr to a SpanSet it is possible to wrap the
     * unique_ptr in a std::move when calling the constructor of a Bootprint and
     * the compiler will implicitly turn the unique_ptr into a shared_ptr and use
     * that to construct the Bootprint.
     */
    explicit Bootprint(std::shared_ptr<geom::SpanSet> inputSpans,
                       geom::Box2I const & region=geom::Box2I());
    explicit Bootprint(std::shared_ptr<geom::SpanSet> inputSpans,
                       afw::table::Schema const & peakSchema,
                       geom::Box2I const & region=geom::Box2I());

    Bootprint(Bootprint const & other) = default;
    Bootprint(Bootprint && ) = default;

    Bootprint & operator=(Bootprint const & other) = default;
    Bootprint & operator=(Bootprint &&) = default;

    virtual ~Bootprint() {}

    /** @brief Indicates if this object is a HeavyBootprint
     */
    virtual bool isHeavy() const { return false; }

    /** @brief Return a shared pointer to the SpanSet
      */
    std::shared_ptr<geom::SpanSet> const & getSpans() const { return _spans;}

    /** @brief Sets the shared pointer to the SpanSet in the Bootprint
     *
     * @param otherSpanSet - Shared pointer to a SpanSet
     */
    void setSpans(std::shared_ptr<geom::SpanSet> otherSpanSet);

    /**
     * @brief Return the Peaks contained in this Bootprint
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
     * This function will throw an error if the PeakCatalog is not empty
     *
     * @param peakSchema - The schema to use in the PeakCatalog
     */
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
     * @brief Return the number of pixels in this Bootprint
     *
     * This function returns the real number of pixels, not the area of the bbox.
     */
    size_t getArea() const { return _spans->getArea(); }

    /**
     * @brief Return the Bootprint's centroid
     *
     * The centroid is calculated as the mean of the pixel centers
     */
    geom::Point2D getCentroid() const { return _spans->computeCentroid(); }

    /**
     * @brief Return the Bootprint's shape (interpreted as an ellipse)
     *
     * The shape is determined by measuring the moments of the pixel
     * centers about its centroid (cf. getCentroid)
     */
    geom::ellipses::Quadrupole getShape() const { return _spans->computeShape(); }

    /**
     * @brief Shift a Bootprint by <tt>(dx, dy)</tt>
     *
     * @param dx - How much to move Bootprint in column direction
     * @param dy - How much to move in row direction
     */
    void shift(int dx, int dy);

    /**
    * @brief Shift a Bootprint by a given extent
    *
    * @param d - ExtentI object which gives the dimensions the Bootprint should be shifted
    */
    void shift(geom::ExtentI const & d) {shift(d.getX(), d.getY());}

    /**
     * @brief Return the Bootprint's bounding box
     */
    geom::Box2I getBBox() const { return _spans->getBBox(); }

    /**
     * @brief Return the corners of the MaskedImage the footprints live in
     */
    geom::Box2I const & getRegion() const { return _region; }

    /**
     * @brief Set the corners of the MaskedImage wherein the footprints dwell
     *
     * @param region - A box describing the corners of the Image the Bootprint derives from
     */
    void setRegion(geom::Box2I const & region) { _region = region; }


    /**
     * @brief Clip the Bootprint such that all values lie inside the supplied Bounding Box
     *
     * @param bbox - Integer box object that defines the boundarys the footprint should be
     *               clipped to.
     */
    void clipTo(geom::Box2I const & bbox);

    /**
     * @brief Tests if a pixel postion falls inside the Bootprint
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
    std::unique_ptr<Bootprint> transform(
        std::shared_ptr<image::Wcs> source,
        std::shared_ptr<image::Wcs> target,
        geom::Box2I const & region,
        bool doClip=true
    ) const;

    /**
     * @brief Reports if this object is persistable
     */
    bool isPersistable() const { return true; }

    /**
     * @brief Dilate the Bootprint with a defined kernel
     *
     * This function enlarges the SpanSet which defines the area of the Bootprint by
     * an amount governed by in input kernel
     *
     * @param r - The radius of the stencil object used to create a dilation kernel
     * @param s - The stencil object used to create the dilation kernel
     */
    void dilate(int r, geom::Stencil s = geom::Stencil::CIRCLE);

    /**
     * @brief Dilate the Bootprint with a defined kernel
     *
     * This function enlarges the SpanSet which defines the area of the Bootprint by
     * an amount governed by the input kernel
     *
     * @param other - SpanSet to use as the kernel in dilation
     */
    void dilate(geom::SpanSet const & other);

    /**
     * @brief Erode the Bootprint with a defined kernel
     *
     * This function reduces the size of the SpanSet which defines the area of the Bootprint
     * by an amount governed by the input kernel
     *
     * @param r - The radius of the stencil object used to create a erosion kernel
     * @param s - The stencil object used to create the erosion kernel
     */
    void erode(int r, geom::Stencil s = geom::Stencil::CIRCLE);

    /**
     * @brief Erode the Bootprint with a defined kernel
     *
     * This function reduces the size of the SpanSet which defines the area of the Bootprint
     * by an amount governed by the input kernel
     *
     * @param other - SpanSet to use as the kernel in erosion
     */
    void erode(geom::SpanSet const & other);

    /**
     * @brief Remove peaks from the PeakCatlog that fall ouside the area of the Bootprint
     */
    void removeOrphanPeaks();

    /**
     * @brief Reports if the Bootprint is simply connected or has multiple components
     */
    bool isContiguous() const { return getSpans()->isContiguous(); };

    /**
    * @brief Split a multi-component Bootprint into a vector of contiguous Bootprints
    *
    * Split a multi-component Bootprint such that each Bootprint in the output vector
    * is contiguous and contains only peaks that can be found within the bounds of the
    * Bootprint
    */
    std::vector<std::unique_ptr<Bootprint>> split() const;

    /**
    * @brief equality operator
    *
    * @param other - The Bootprint for which equality will be computed
    */
    bool operator==(Bootprint const & other) const;

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
     * Write an instance of a Bootprint to an output Archive
     */
    void write(OutputArchiveHandle & handle) const override;

    friend class BootprintFactory;

 private:

    friend class BootprintMerge;

    /*
     * Static method used to unpersist the SpanSet member of the Bootprint class
     */
    static std::unique_ptr<Bootprint> readSpanSet(afw::table::BaseCatalog const &,
                                                  afw::table::io::InputArchive const &);
    /*
     * Static method used to unpersit the PeakCatalog member of the Bootprint class
     */
    static void readPeaks(afw::table::BaseCatalog const &, Bootprint &);

    std::shared_ptr<geom::SpanSet> _spans;    //!< The SpanSet representing area on image
    PeakCatalog _peaks;                 //!< The peaks lying in this footprint
    geom::Box2I _region;     //!< The corners of the MaskedImage the footprints live in
};

}}} // Close namespace lsst::afw::detection

#endif // LSST_DETECTION_BOOTPRINT_H
