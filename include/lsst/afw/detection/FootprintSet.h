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
/*
 * Represent a collections of footprints associated with image data
 */
#include <cstdint>

#include "lsst/geom.h"
#include "lsst/afw/detection/Threshold.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/detection/FootprintCtrl.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/table/Source.h"

namespace lsst {
namespace afw {
namespace detection {

/** Pixel type for FootprintSet::insertIntoImage()
 *
 * This is independent of the template parameters for FootprintSet, and
 * including it within FootprintSet makes it difficult for SWIG to interpret
 * the type.
 */
typedef std::uint64_t FootprintIdPixel;

/**
 * A set of Footprints, associated with a MaskedImage
 */
class FootprintSet {
public:
    /// The FootprintSet's set of Footprint%s
    typedef std::vector<std::shared_ptr<Footprint>> FootprintList;

    /**
     * Find a FootprintSet given an Image and a threshold
     *
     * @param img Image to search for objects
     * @param threshold threshold to find objects
     * @param npixMin minimum number of pixels in an object
     * @param setPeaks should I set the Peaks list?
     */
    template <typename ImagePixelT>
    FootprintSet(image::Image<ImagePixelT> const& img, Threshold const& threshold, int const npixMin = 1,
                 bool const setPeaks = true);

    /**
     * Find a FootprintSet given a Mask and a threshold
     *
     * @param img Image to search for objects
     * @param threshold threshold to find objects
     * @param npixMin minimum number of pixels in an object
     */
    template <typename MaskPixelT>
    FootprintSet(image::Mask<MaskPixelT> const& img, Threshold const& threshold, int const npixMin = 1);

    /**
     * Find a FootprintSet given a MaskedImage and a threshold
     *
     * Go through an image, finding sets of connected pixels above threshold
     * and assembling them into Footprint%s;  the resulting set of objects
     * is returned
     *
     * If threshold.getPolarity() is true, pixels above the Threshold are
     * assembled into Footprints; if it's false, then pixels *below* Threshold
     * are processed (Threshold will probably have to be below the background level
     * for this to make sense, e.g. for difference imaging)
     *
     * @param img MaskedImage to search for objects
     * @param threshold threshold for footprints (controls size)
     * @param planeName mask plane to set (if != "")
     * @param npixMin minimum number of pixels in an object
     * @param setPeaks should I set the Peaks list?
     */
    template <typename ImagePixelT, typename MaskPixelT>
    FootprintSet(image::MaskedImage<ImagePixelT, MaskPixelT> const& img, Threshold const& threshold,
                 std::string const& planeName = "", int const npixMin = 1, bool const setPeaks = true);

    /**
     * Construct an empty FootprintSet given a region that its footprints would have lived in
     *
     * @param region the desired region
     */
    FootprintSet(lsst::geom::Box2I region);
    /**
     * Copy constructor
     *
     * @param rhs the input FootprintSet
     */
    FootprintSet(FootprintSet const& rhs);
    FootprintSet(FootprintSet const& set, int rGrow, FootprintControl const& ctrl);
    FootprintSet(FootprintSet&& rhs);
    ~FootprintSet();
    /**
     * Grow all the Footprints in the input FootprintSet, returning a new FootprintSet
     *
     * The output FootprintSet may contain fewer Footprints, as some may well have been merged
     *
     * @param set the input FootprintSet
     * @param rGrow Grow Footprints by r pixels
     * @param isotropic Grow isotropically (as opposed to a Manhattan metric)
     *
     * @note Isotropic grows are significantly slower
     */
    FootprintSet(FootprintSet const& set, int rGrow, bool isotropic = true);
    /**
     * Return the FootprintSet corresponding to the merge of two input FootprintSets
     *
     * @todo Implement this.  There's RHL Pan-STARRS code to do it, but it isn't yet converted to LSST C++
     */
    FootprintSet(FootprintSet const& footprints1, FootprintSet const& footprints2, bool const includePeaks);

    /// Assignment operator.
    FootprintSet& operator=(FootprintSet const& rhs);
    FootprintSet& operator=(FootprintSet&& rhs);

    void swap(FootprintSet& rhs) noexcept {
        using std::swap;  // See Meyers, Effective C++, Item 25
        swap(*_footprints, *rhs.getFootprints());
        lsst::geom::Box2I rhsRegion = rhs.getRegion();
        rhs.setRegion(getRegion());
        setRegion(rhsRegion);
    }

    void swapFootprintList(FootprintList& rhs) noexcept {
        using std::swap;
        swap(*_footprints, rhs);
    }

    /**:
     * Return the Footprint%s of detected objects
     */
    std::shared_ptr<FootprintList> getFootprints() { return _footprints; }

    /**:
     * Set the Footprint%s of detected objects
     */
    void setFootprints(std::shared_ptr<FootprintList> footprints) { _footprints = footprints; }

    /**
     * Retun the Footprint%s of detected objects
     */
    std::shared_ptr<FootprintList const> const getFootprints() const { return _footprints; }

    /**
     *  Add a new record corresponding to each footprint to a SourceCatalog.
     *
     *  @param[in,out]  catalog     Catalog to append new sources to.
     *
     *  The new sources will have their footprints set to point to the footprints in the
     *  footprint set; they will not be deep-copied.
     */
    void makeSources(afw::table::SourceCatalog& catalog) const;

    /**
     * Set the corners of the FootprintSet's MaskedImage to region
     *
     * @param region desired region
     *
     * @note updates all the Footprints' regions too
     */
    void setRegion(lsst::geom::Box2I const& region);

    /**
     * Return the corners of the MaskedImage
     */
    lsst::geom::Box2I const getRegion() const { return _region; }

    /**
     * Return an Image with pixels set to the Footprint%s in the FootprintSet
     *
     * @returns an std::shared_ptr<image::Image>
     */
    std::shared_ptr<image::Image<FootprintIdPixel>> insertIntoImage() const;

    template <typename MaskPixelT>
    void setMask(image::Mask<MaskPixelT>* mask,  ///< Set bits in the mask
                 std::string const& planeName    ///< Here's the name of the mask plane to fit
    ) {
        for (auto const& foot : *_footprints) {
            foot->getSpans()->setMask(*mask, image::Mask<MaskPixelT>::getPlaneBitMask(planeName));
        }
    }

    template <typename MaskPixelT>
    void setMask(std::shared_ptr<image::Mask<MaskPixelT>> mask,  ///< Set bits in the mask
                 std::string const& planeName                    ///< Here's the name of the mask plane to fit
    ) {
        setMask(mask.get(), planeName);
    }

    /**
     * Merge a FootprintSet into *this
     *
     * @param rhs the Footprints to merge
     * @param tGrow No. of pixels to grow this Footprints
     * @param rGrow No. of pixels to grow rhs Footprints
     * @param isotropic Use (expensive) isotropic grow
     */
    void merge(FootprintSet const& rhs, int tGrow = 0, int rGrow = 0, bool isotropic = true);

    /**
     * Convert all the Footprints in the FootprintSet to be HeavyFootprint%s
     *
     * @param mimg the image providing pixel values
     * @param ctrl Control how we manipulate HeavyFootprints
     */
    template <typename ImagePixelT, typename MaskPixelT>
    void makeHeavy(image::MaskedImage<ImagePixelT, MaskPixelT> const& mimg,
                   HeavyFootprintCtrl const* ctrl = NULL);

private:
    std::shared_ptr<FootprintList> _footprints;  ///< the Footprints of detected objects
    lsst::geom::Box2I _region;  ///< The corners of the MaskedImage that the detections live in
};
}  // namespace detection
}  // namespace afw
}  // namespace lsst

#endif
