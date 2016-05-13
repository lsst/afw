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
/**
 * \file
 * \brief Represent a collections of footprints associated with image data
 */
#include <cstdint>

#include "lsst/afw/geom.h"
#include "lsst/afw/detection/Threshold.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/detection/FootprintCtrl.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/table/Source.h"

namespace lsst {
namespace afw {
namespace detection {

/// Pixel type for FootprintSet::insertIntoImage()
///
/// This is independent of the template parameters for FootprintSet, and
/// including it within FootprintSet makes it difficult for SWIG to interpret
/// the type.
typedef std::uint64_t FootprintIdPixel;

/************************************************************************************************************/
/*!
 * \brief A set of Footprints, associated with a MaskedImage
 *
 */
class FootprintSet : public lsst::daf::base::Citizen {
public:

    /// The FootprintSet's set of Footprint%s
    typedef std::vector<Footprint::Ptr> FootprintList;

#ifndef SWIG
    template <typename ImagePixelT>
    FootprintSet(image::Image<ImagePixelT> const& img,
                 Threshold const& threshold,
                 int const npixMin=1, bool const setPeaks=true);

    template <typename MaskPixelT>
    FootprintSet(image::Mask<MaskPixelT> const& img,
                 Threshold const& threshold,
                 int const npixMin=1);

    template <typename ImagePixelT, typename MaskPixelT>
    FootprintSet(image::MaskedImage<ImagePixelT, MaskPixelT> const& img,
                 Threshold const& threshold,
                 std::string const& planeName = "",
                 int const npixMin=1, bool const setPeaks=true);

#else // workaround for https://github.com/swig/swig/issues/245
    // if that bug is fixed then you may update footprintset.i by uncommenting two lines
    // and removing this section. However, you must continue to provide SWIG
    // the alternate version of the template <typename MaskPixelT> constructor, because
    // SWIG cannot disambiguate that from the template <typename ImagePixelT> constructor.

    FootprintSet(image::Mask<image::MaskPixel> const& img,
                 Threshold const& threshold,
                 int const npixMin=1);

    FootprintSet(image::Image<std::uint16_t> const& img,
                 Threshold const& threshold,
                 int const npixMin=1, bool const setPeaks=true);
    FootprintSet(image::MaskedImage<std::uint16_t, image::MaskPixel> const& img,
                 Threshold const& threshold,
                 std::string const& planeName = "",
                 int const npixMin=1, bool const setPeaks=true);

    FootprintSet(image::Image<int> const& img,
                 Threshold const& threshold,
                 int const npixMin=1, bool const setPeaks=true);
    FootprintSet(image::MaskedImage<int, image::MaskPixel> const& img,
                 Threshold const& threshold,
                 std::string const& planeName = "",
                 int const npixMin=1, bool const setPeaks=true);

    FootprintSet(image::Image<float> const& img,
                 Threshold const& threshold,
                 int const npixMin=1, bool const setPeaks=true);
    FootprintSet(image::MaskedImage<float, image::MaskPixel> const& img,
                 Threshold const& threshold,
                 std::string const& planeName = "",
                 int const npixMin=1, bool const setPeaks=true);

    FootprintSet(image::Image<double> const& img,
                 Threshold const& threshold,
                 int const npixMin=1, bool const setPeaks=true);
    FootprintSet(image::MaskedImage<double, image::MaskPixel> const& img,
                 Threshold const& threshold,
                 std::string const& planeName = "",
                 int const npixMin=1, bool const setPeaks=true);

#endif

    FootprintSet(geom::Box2I region);
    FootprintSet(FootprintSet const&);
    FootprintSet(FootprintSet const& set, int rGrow, FootprintControl const& ctrl);
    FootprintSet(FootprintSet const& set, int rGrow, bool isotropic=true);
    FootprintSet(FootprintSet const& footprints1, 
                 FootprintSet const& footprints2,
                 bool const includePeaks);

    FootprintSet& operator=(FootprintSet const& rhs);

    void swap(FootprintSet& rhs) {
        using std::swap;                    // See Meyers, Effective C++, Item 25        
        swap(*_footprints, *rhs.getFootprints());
        geom::Box2I rhsRegion = rhs.getRegion();
        rhs.setRegion(getRegion());
        setRegion(rhsRegion);
    }

    void swapFootprintList(FootprintList& rhs) {
        using std::swap;
        swap(*_footprints, rhs);
    }

    /**:
     * Return the Footprint%s of detected objects
     */
    PTR(FootprintList) getFootprints() { return _footprints; } 

    /**:
     * Set the Footprint%s of detected objects
     */
    void setFootprints(PTR(FootprintList) footprints) { _footprints = footprints; }

    /**
     * Retun the Footprint%s of detected objects
     */
    CONST_PTR(FootprintList) const getFootprints() const { return _footprints; }
    
    /**
     *  @brief Add a new record corresponding to each footprint to a SourceCatalog.
     *
     *  @param[in,out]  catalog     Catalog to append new sources to.
     *
     *  The new sources will have their footprints set to point to the footprints in the
     *  footprint set; they will not be deep-copied.
     */
    void makeSources(afw::table::SourceCatalog & catalog) const;

    void setRegion(geom::Box2I const& region);

    /**
     * Return the corners of the MaskedImage
     */
    geom::Box2I const getRegion() const { return _region; } 

    PTR(image::Image<FootprintIdPixel>) insertIntoImage(
        const bool relativeIDs
        ) const;

    template <typename MaskPixelT>
    void setMask(
        image::Mask<MaskPixelT> *mask, ///< Set bits in the mask
        std::string const& planeName   ///< Here's the name of the mask plane to fit
    ) {
        setMaskFromFootprintList(
            mask, 
            _footprints,                // calling getFootprints() confuses clang++ 3.0 and leaks memory
            image::Mask<MaskPixelT>::getPlaneBitMask(planeName)
        );        
    }

    template <typename MaskPixelT>
    void setMask(
        PTR(image::Mask<MaskPixelT>) mask, ///< Set bits in the mask
        std::string const& planeName   ///< Here's the name of the mask plane to fit
    ) {
        setMask(mask.get(), planeName);
    }

    void merge(FootprintSet const& rhs, int tGrow=0, int rGrow=0, bool isotropic=true);

    template <typename ImagePixelT, typename MaskPixelT>
    void makeHeavy(image::MaskedImage<ImagePixelT, MaskPixelT> const& mimg,
                   HeavyFootprintCtrl const* ctrl=NULL
                  );
private:
    std::shared_ptr<FootprintList> _footprints;        //!< the Footprints of detected objects
    geom::Box2I _region;                //!< The corners of the MaskedImage that the detections live in
};

}}}

#endif
