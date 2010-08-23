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
 
#if !defined(LSST_AFW_MATH_OFFSETIMAGE_H)
#define LSST_AFW_MATH_OFFSETIMAGE_H 1

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Statistics.h"
#include "lsst/afw/math/warpExposure.h"

namespace lsst {
namespace afw {
namespace math {

template<typename ImageT>
typename ImageT::Ptr offsetImage(ImageT const& image, float dx, float dy, std::string const& algorithmName="lanczos5");
template<typename ImageT>
typename ImageT::Ptr rotateImageBy90(ImageT const& image, int nQuarter);

template<typename ImageT>
typename ImageT::Ptr flipImage(ImageT const& inImage, ///< The %image to flip
                               bool flipLR,           ///< Flip left <--> right?
                               bool flipTB            ///< Flip top <--> bottom?
                              );
template<typename ImageT>
typename ImageT::Ptr binImage(ImageT const& inImage, int const binsize,
                              lsst::afw::math::Property const flags=lsst::afw::math::MEAN);

    
}}}
#endif
