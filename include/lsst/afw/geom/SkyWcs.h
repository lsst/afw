// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2017 AURA/LSST.
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

#ifndef LSST_AFW_GEOM_SKYWCS_H
#define LSST_AFW_GEOM_SKYWCS_H

#include <memory>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "astshim.h"
#include "ndarray.h"

#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Endpoint.h"
#include "lsst/afw/geom/Transform.h"
#include "lsst/daf/base/PropertyList.h"

namespace lsst {
namespace afw {
namespace geom {

/**
Make a WCS CD matrix

@param[in] scale  Pixel scale as an angle on sky/pixels
@param[in] orientation  Position angle of focal plane +Y, measured from N through E.
                        At 0 degrees, +Y is along N and +X is along E/W if flipX false/true
                        At 90 degrees, +Y is along E and +X is along S/N if flipX false/true
@param[in] flipX  Fip x axis (see orientation for details)?

@return the CD matrix, where element (i-1, j-1) corresponds to FITS keyword CDi_j
                        and i, j have range [1, 2]

*/
Eigen::Matrix2d makeCdMatrix(Angle const &scale, Angle const &orientation = 0 * degrees, bool flipX = false);

/**
A WCS that transform pixels to ICRS RA/Dec using the LSST standard for pixels

@anchor pixel_position_standards
The LSST standard for pixel position is: 0,0 is the center of the lower left pixel of the parent image
(unlike FITS standard, which uses 1,1 as the center of the lower left pixel of the subimage).

The forward direction transforms from pixels (actual, if known, else nominal) to ICRS RA, Dec.

@anchor skywcs_frames The contained ast::FrameSet must have the following frames.
All are of class ast::Frame except the sky frame. The domain is listed first, if set:
- ACTUAL_PIXEL0 (optional): actual pixel position using the @ref pixel_position_standards "LSST standard";
    The 0 is to remind users that this frame is 0-based.
    Actual pixels include effects such as "tree ring" distortions and electrical effects at the edge of CCDs.
    This frame should only be provided if there is a reasonable model for these imperfections.
- PIXEL0: nominal pixel position, using the @ref pixel_position_standards "LSST standard";
    The 0 is to remind users that this frame is 0-based.
    Nominal pixels are rectangular; the positions have been compensated for effects such as "tree ring"
    distortions (if an ACTUAL_PIXEL0 frame was provided) or such effects are assumed negligible (if not).
- GRID: nominal pixel position, using the @ref pixel_position_standards "FITS standard".
- IWC: intermediate world coordinates (the FITS WCS concept).
- An ast::SkyFrame with System=ICRS and standard axis order RA, Dec.

If ACTUAL_PIXEL0 is present then it will be the base frame (the starting point for forward transformations
else PIXEL0 will be the base frame. The SkyFrame is always the current frame (the ending point for
forward transformations).

Constructors must set the following properties of the contained ast::FrameSet:
- Include the frames listed above, with specified domains.
    Note that AST will provide the IWC frame when reading standard WCS FITS
    data using an ast::FitsChan if you set "IWC=1" as a property of ast::FitsChan.
    Other constructors may have to work harder.
- For the output ast::SkyFrame set SkyRef=CRVAL (in radians) and SkyRefIs="Ignored".
    SkyWcs uses this to look up CRVAL; "Ignored" is required to prevent CRVAL from being used as an offset
    from the desired WCS.
*/
class SkyWcs : public TransformPoint2ToIcrsCoord {
public:
    SkyWcs(SkyWcs const &) = default;
    SkyWcs(SkyWcs &&) = default;
    SkyWcs &operator=(SkyWcs const &) = delete;
    SkyWcs &operator=(SkyWcs &&) = delete;

    /**
    Construct a pure tangent SkyWcs

    @param[in] crpix  Center of projection on the CCD using the LSST convention:
                        0, 0 is the lower left pixel of the parent image
    @param[in] crval  Center of projection on the sky
    @param[in] cdMatrix  CD matrix, where element (i-1, j-1) corresponds to FITS keyword CDi_j
                        and i, j have range [1, 2]. May be computed by calling makeCdMatrix.
    */
    explicit SkyWcs(Point2D const &crpix, coord::IcrsCoord const &crval, Eigen::Matrix2d const &cdMatrix);

    /**
    Construct a WCS from FITS keywords

    In addition to standard FITS WCS keywords uses these IRAF keywords:
    LTV1, LTV2: offset of subimage with respect to parent image

    @param[in] metadata  FITS header metadata
    @param[in] strip  Strip keywords used to create the SkyWcs?
                Even if true. keywords that might used for other purposes are retained,
                including LTV1, LTV2 and all date and time keywords except EQUINOX
    */
    explicit SkyWcs(daf::base::PropertyList &metadata, bool strip = true);

    /**
    Construct a WCS from an ast::FrameSet

    This is the most general constructor; it can be used to define any celestial WCS
    as long as it contains the @ref skywcs_frames "required frames".

    @throw lsst::pex::exceptions::InvalidParameterError if `frameSet` is missing
    any of the @ref skywcs_frames "required frames".
    */
    explicit SkyWcs(ast::FrameSet const &frameSet);

    ~SkyWcs(){};

    /**
    Get the pixel scale at the specified pixel position

    The scale is the square root of the area of the specified pixel on the sky.
    */
    Angle getPixelScale(Point2D const &pixel) const;

    /**
    Get the pixel scale at the pixel origin

    The scale is the square root of the area of the specified pixel on the sky.
    */
    Angle getPixelScale() const;

    /**
    Get CRPIX, the pixel origin, using the LSST convention
    */
    Point2D getPixelOrigin() const;

    /**
    Get CRVAL, the sky origin or celestial fiducial point
    */
    coord::IcrsCoord getSkyOrigin() const;

    /**
    Get the 2x2 CD matrix at the specified pixel position
    */
    Eigen::Matrix2d getCdMatrix(Point2D const &pixel) const;

    /**
    Get the 2x2 CD matrix at the pixel origin
    */
    Eigen::Matrix2d getCdMatrix() const;

    /**
    Get a local TAN WCS approximation to this WCS at the specified pixel position
    */
    std::shared_ptr<SkyWcs> getTanWcs(Point2D const &pixel) const;

    /**
    Return a copy of this SkyWcs with the pixel origin by the specified amount
    */
    std::shared_ptr<SkyWcs> copyAtShiftedPixelOrigin(Extent2D const &shift) const;

    /**
    Compute sky position(s) from pixel position(s)

    This is another name for applyForward, plus an overload that takes a pair of doubles.
    */
    //@{
    std::pair<Angle, Angle> pixelToSky(double x, double y) const;
    coord::IcrsCoord pixelToSky(Point2D const &pixel) const { return applyForward(pixel); };
    std::vector<coord::IcrsCoord> pixelToSky(std::vector<Point2D> const &pixels) const {
        return applyForward(pixels);
    }
    //@}

    /**
    Compute pixel position(s) from sky position(s)

    This is another name for applyInverse, plus an overload that takes a pair of lsst:afw::geom::Angle.
    */
    //@{
    std::pair<double, double> skyToPixel(Angle const &ra, Angle const &dec) const;
    Point2D skyToPixel(coord::IcrsCoord const &sky) const { return applyInverse(sky); }
    std::vector<Point2D> skyToPixel(std::vector<coord::IcrsCoord> const &sky) const {
        return applyInverse(sky);
    }
    //@}

    static std::string getShortClassName();

    /**
     * Deserialize a SkyWcs from an input stream
     *
     * @param[in] is  input stream from which to deserialize this SkyWcs
     */
    static SkyWcs readStream(std::istream & is);

    /// Deserialize a SkyWcs from a string, using the same format as readStream
    static SkyWcs readString(std::string & str);

    /**
     * Serialize this SkyWcs to an output stream
     *
     * Version 1 format is as follows:
     * - The version number (an integer)
     * - A space
     * - The short class name, as obtained from getShortClassName
     * - A space
     * - The contained ast::FrameSet written using FrameSet.show(os, false)
     *
     * @param[out] os  output stream to which to serialize this SkyWcs
     */
    void writeStream(std::ostream & os) const override;

    /// Serialize this SkyWcs to a string, using the same format as writeStream
    std::string writeString() const override;

private:
    // Construct a SkyWcs from a shared pointer to an ast::FrameSet
    explicit SkyWcs(std::shared_ptr<ast::FrameSet> &&frameSet);

    // Check a FrameSet to see if it can safely be used for a SkyWcs
    // Return a copy so that it can be used as an argument to the SkyWcs(shared_ptr<FrameSet>) constructor
    std::shared_ptr<ast::FrameSet> _checkFrameSet(ast::FrameSet const &frameSet) const;
};

/**
 * A Transform obtained by putting two SkyWcs objects "back to back".
 *
 * @param src the WCS for the source pixels
 * @param dst the WCS for the destination pixels
 * @returns a Transform whose forward transformation converts from `src`
 *          pixels to `dst` pixels, and whose inverse transformation converts
 *          in the opposite direction.
 *
 * @exceptsafe Provides basic exception safety.
 *
 * @note Parameters are provided in the order `dst`, `src` for consistency with
 *       XYTransformFromWcsPair.
 */
TransformPoint2ToPoint2 makeWcsPairTransform(SkyWcs const &src, SkyWcs const &dst);

}  // namespace geom
}  // namespace afw
}  // namespace lsst

#endif
