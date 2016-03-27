// -*- lsst-c++ -*-
/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
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
     * @param[in] pixelsToTanPixels  an XYTransform that converts from PIXELS to TAN_PIXELS coordinates
     *              in the forward direction. This can be obtained from an exposure using:
     *                  detector = exposure.getDetector()
     *                  pixelsToTanPixels = detector.getTransformMap()[lsst.afw.cameraGeom.TAN_PIXELS]
     *
     * @throw pex::exceptions::InvalidParameterError if tanWcs.hasDistortion()
     */
    DistortedTanWcs(
        TanWcs const &tanWcs,
        geom::XYTransform const &pixelsToTanPixels
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

    /// @warning not implemented
    virtual void shiftReferencePixel(double dx, double dy);

    bool isPersistable() const { return false; }

    bool hasDistortion() const { return true; }

    /// return the pure tan WCS component
    PTR(Wcs) getTanWcs() const { return TanWcs::clone(); }

    /// return the PIXELS to TAN_PIXELS XYTransform
    PTR(geom::XYTransform) getPixelToTanPixel() const { return _pixelsToTanPixelsPtr->clone(); }

protected:

    /**
    Worker routine for skyToPixel

    @param[in] sky1  sky position, longitude (e.g. RA)
    @param[in] sky2  sky position, latitude (e.g. dec)
    */
    virtual void pixelToSkyImpl(double pixel1, double pixel2, geom::Angle skyTmp[2]) const;

    /**
    Worker routine for pixelToSky

    @param[in] pixel1  pixel position, x
    @param[in] pixel2  pixel position, y
    @param[out] sky  sky position (longitude, latitude, e.g. RA, Dec)
    */
    virtual geom::Point2D skyToPixelImpl(geom::Angle sky1, geom::Angle sky2) const;

private:

    PTR(geom::XYTransform) _pixelsToTanPixelsPtr;   // XYTransform that converts from PIXELS to TAN_PIXELS 
                                                    // coordinates in the forward direction

};

}}} // namespace lsst::afw::image

#endif
