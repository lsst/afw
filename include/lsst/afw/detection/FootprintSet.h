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
#include "lsst/afw/geom.h"
#include "lsst/afw/detection/Threshold.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/image/MaskedImage.h"

namespace lsst {
namespace afw {
namespace detection {

/************************************************************************************************************/
/*!
 * \brief A set of Footprints, associated with a MaskedImage
 *
 */
template<typename ImagePixelT, typename MaskPixelT=lsst::afw::image::MaskPixel>
class FootprintSet : public lsst::daf::data::LsstBase {
public:
    typedef boost::shared_ptr<FootprintSet> Ptr;
    /// The FootprintSet's set of Footprint%s
    typedef std::vector<Footprint::Ptr> FootprintList;

    FootprintSet(image::Image<ImagePixelT> const& img,
                 Threshold const& threshold,
                 double const includeThresholdMultiplier=1.0,
                 int const npixMin=1, bool const setPeaks=true);
    FootprintSet(image::Mask<MaskPixelT> const& img,
                 Threshold const& threshold,
                 double const includeThresholdMultiplier=1.0,
                 int const npixMin=1);
    FootprintSet(image::MaskedImage<ImagePixelT, MaskPixelT> const& img,
                 Threshold const& threshold,
                 double const includeThresholdMultiplier,
                 std::string const& planeName = "",
                 int const npixMin=1, bool const setPeaks=true);
    FootprintSet(image::MaskedImage<ImagePixelT, MaskPixelT> const& img,
                 Threshold const& threshold,
                 std::string const& planeName = "",
                 int const npixMin=1, bool const setPeaks=true);
    FootprintSet(image::MaskedImage<ImagePixelT, MaskPixelT> const& img,
                 Threshold const& threshold,
                 int x,
                 int y,
                 std::vector<PTR(Peak)> const* peaks = NULL);
    FootprintSet(geom::Box2I region);
    FootprintSet(FootprintSet const&);
    FootprintSet(FootprintSet const& set, int rGrow, bool isotropic=true);
    FootprintSet(FootprintSet const& footprints1, 
                 FootprintSet const& footprints2,
                 bool const includePeaks);
    ~FootprintSet();

    FootprintSet& operator=(FootprintSet const& rhs);

    template<typename RhsImagePixelT, typename RhsMaskPixelT>
    void swap(FootprintSet<RhsImagePixelT, RhsMaskPixelT>& rhs) {
        using std::swap;                    // See Meyers, Effective C++, Item 25
        
        swap(*_footprints, *rhs.getFootprints());
        geom::Box2I rhsRegion = rhs.getRegion();
        rhs.setRegion(getRegion());
        setRegion(rhsRegion);
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
    
    void setRegion(geom::Box2I const& region);
    /**
     * Return the corners of the MaskedImage
     */
    geom::Box2I const getRegion() const { return _region; } 

    typename image::Image<boost::uint16_t>::Ptr insertIntoImage(
        const bool relativeIDs
    );
    void setMask(
        image::Mask<MaskPixelT> *mask, ///< Set bits in the mask
        std::string const& planeName   ///< Here's the name of the mask plane to fit
    ) {
        setMaskFromFootprintList(
            mask, 
            getFootprints(),
            image::Mask<MaskPixelT>::getPlaneBitMask(planeName)
        );        
    }

    void setMask(
        typename image::Mask<MaskPixelT>::Ptr mask, ///< Set bits in the mask
        std::string const& planeName   ///< Here's the name of the mask plane to fit
    ) {
        setMask(mask.get(), planeName);
    }

    void merge(FootprintSet const& rhs, int tGrow=0, int rGrow=0, bool isotropic=true);
private:
    void findFootprintsAndMask(const image::MaskedImage<ImagePixelT, MaskPixelT> &maskedImg,
                               Threshold const &threshold, double const includeThresholdMultiplier,
                               std::string const &planeName, int const npixMin, bool const setPeaks);

    boost::shared_ptr<FootprintList> _footprints;        //!< the Footprints of detected objects
    geom::Box2I _region;                //!< The corners of the MaskedImage that the detections live in
};

template<typename ImagePixelT, typename MaskPixelT>
typename FootprintSet<ImagePixelT>::Ptr makeFootprintSet(
        image::Image<ImagePixelT> const& img,
        Threshold const& threshold,
        double const includeThresholdMultiplier=1.0,
        std::string const& = "",
        int const npixMin=1
) {
    return typename FootprintSet<ImagePixelT, MaskPixelT>::Ptr(
        new FootprintSet<ImagePixelT, MaskPixelT>(img, threshold, includeThresholdMultiplier, npixMin)
    );
}

template<typename MaskPixelT>
typename FootprintSet<MaskPixelT, MaskPixelT>::Ptr makeFootprintSet(
        image::Mask<MaskPixelT> const& msk,
        Threshold const& threshold,
        double const includeThresholdMultiplier=1.0,
        std::string const& = "",
        int const npixMin=1
) {
    return typename FootprintSet<MaskPixelT, MaskPixelT>::Ptr(
        new FootprintSet<MaskPixelT, MaskPixelT>(msk, threshold, includeThresholdMultiplier, npixMin)
    );
}

template<typename ImagePixelT, typename MaskPixelT>
typename FootprintSet<ImagePixelT, MaskPixelT>::Ptr makeFootprintSet(
        image::MaskedImage<ImagePixelT, MaskPixelT> const& img,
        Threshold const& threshold,
        double const includeThresholdMultiplier,
        std::string const& planeName = "",
        int const npixMin=1
) {
    return typename FootprintSet<ImagePixelT, MaskPixelT>::Ptr(
        new FootprintSet<ImagePixelT, MaskPixelT>(
            img, threshold, includeThresholdMultiplier, planeName, npixMin
        )
    );
}

template<typename ImagePixelT, typename MaskPixelT>
typename FootprintSet<ImagePixelT, MaskPixelT>::Ptr makeFootprintSet(
        image::MaskedImage<ImagePixelT, MaskPixelT> const& img,
        Threshold const& threshold,
        std::string const& planeName = "",
        int const npixMin=1
) {
    return typename FootprintSet<ImagePixelT, MaskPixelT>::Ptr(
        new FootprintSet<ImagePixelT, MaskPixelT>(
            img, threshold, 1.0, planeName, npixMin
        )
    );
}

template<typename ImagePixelT, typename MaskPixelT>
typename FootprintSet<ImagePixelT, MaskPixelT>::Ptr makeFootprintSet(
        image::MaskedImage<ImagePixelT, MaskPixelT> const& img,
        Threshold const& threshold,
        int x,
        int y,
        std::vector<PTR(Peak)> const* peaks = NULL
) {
    return PTR(FootprintSet<ImagePixelT, MaskPixelT>)(
        new FootprintSet<ImagePixelT, MaskPixelT>(img, threshold, x, y, peaks)
    );
}

template<typename ImagePixelT, typename MaskPixelT>
typename FootprintSet<ImagePixelT>::Ptr makeFootprintSet(
        FootprintSet<ImagePixelT, MaskPixelT> const& rhs, //!< the input FootprintSet
        int r,                          //!< Grow Footprints by r pixels
        bool isotropic=true             //!< Grow isotropically (as opposed to a Manhattan metric)
                                        //!< @note Isotropic grows are significantly slower
) {
    return typename detection::FootprintSet<ImagePixelT, MaskPixelT>::Ptr(
        new FootprintSet<ImagePixelT, MaskPixelT>(rhs, r, isotropic)
    );
}

}}}

#endif
