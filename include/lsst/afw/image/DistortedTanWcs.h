// -*- lsst-c++ -*-
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
#ifndef LSST_AFW_IMAGE_DISTORTEDTANWCS_H
#define LSST_AFW_IMAGE_DISTORTEDTANWCS_H

#include "lsst/afw/image/TanWcs.h"
#include "lsst/afw/geom.h"

namespace lsst {
namespace afw {
namespace image {

/**
 *  @brief Combination of a TAN WCS and a distortion model
 *
 * This object represents a common case for raw or minimally processed data; we have estimates for:
 * - a pure TAN WCS based on telescope pointing
 * - a distortion model which converts between PIXELS and TAN_PIXELS coordinates, based on an optical model
 * This object combines the two in a way that makes it easy to estimate sky<->pixel coordinates.
 *
 * @warning This is not a full-fledged WCS class in several respects:
 * - It has no standard FITS WCS representation.
 * - It does not support persistence or unpersistence from FITS. The usual use case is to represent the WCS
 *   for an exposure that has a plain TAN WCS and a distortion model in its detector;
 *   As such, the data is present in two disparate places, making persistence difficult.
 * - It cannot be flipped or rotated. This is probably fixable, but does not seem worth the work.
 * - 
 */
class DistortedTanWcs : public TanWcs {
public:
    /**
     * @brief Construct a DistortedTanWcs
     *
     * @param[in] tanWcs  pure tangent-plane WCS
     * @param[in] pixelToTanPixel  an XYTransform that converts from PIXEL to TAN_PIXEL coordinates
     *              in the forward direction. This can be obtained from an exposure using:
     *                  detector = exposure.getDetector()
     *                  pixelToTanPixel = detector.getTransformMap()[lsst.afw.cameraGeom.TAN_PIXEL]
     *
     * @throw pex::exceptions::InvalidParameterError if tanWcs.hasDistortion()
     */
    DistortedTanWcs(
        TanWcs const &tanWcs,
        geom::XYTransform const &pixelToTanPixel
    );

    virtual ~DistortedTanWcs() {};

    /// Polymorphic deep-copy.
    virtual PTR(Wcs) clone() const;

    /// @warning not implemented (because XYTransform operator== is not implemented)
    bool operator==(Wcs const & other) const;

    /// @warning not implemented
    virtual void flipImage(int flipLR, int flipTB, lsst::afw::geom::Extent2I dimensions) const;

    /// @warning not implemented
    virtual void rotateImageBy90(int nQuarter, lsst::afw::geom::Extent2I dimensions) const;

    bool isPersistable() const { return false; }

    bool hasDistortion() const { return true; }

    /// return the pure tan WCS component
    PTR(Wcs) getTanWcs() const { return TanWcs::clone(); }

    /// return the PIXEL to TAN_PIXEL XYTransform
    PTR(geom::XYTransform) getPixelToTanPixel() const { return _pixelToTanPixelPtr->clone(); }

protected:

    virtual void pixelToSkyImpl(double pixel1, double pixel2, geom::Angle skyTmp[2]) const;
    virtual geom::Point2D skyToPixelImpl(geom::Angle sky1, geom::Angle sky2) const;

private:

    PTR(geom::XYTransform) _pixelToTanPixelPtr;     /// XYTransform that converts from PIXEL to TAN_PIXEL
        /// coordinates in the forward direction

};

}}} // namespace lsst::afw::image

#endif
