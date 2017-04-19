// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * See COPYRIGHT file at the top of the source tree.
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

#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/SpherePoint.h"
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

The LSST standard is: 0,0 is the lower left pixel of the parent image
(unlike FITS, which uses 1,1 as the lower left pixel of the subimage).

The forward direction transforms from pixels (actual, if known, else nominal) to ICRS RA, Dec.

The contained ast::FrameSet has the following frames. All are of class ast::Frame
except the sky frame. The domain is listed first, if set:
- ACTUAL_PIXEL0 (optional): actual pixels in LSST convention: 0,0 is lower left corner of parent image.
    The 0 is to remind users that this frame is 0-based.
- PIXEL0: nominal pixels in LSST convention: 0,0 is lower left corner of parent image
- GRID: nominal pixels in FITS convention: 1,1 is lower left corner of subimage
- IWC: intermediate world coordinates (the FITS WCS concept)
- An ast::SkyFrame with System=ICRS and standard axis order RA, Dec.

If ACTUAL_PIXEL0 is present then it will be the base frame, else PIXEL0 will be the base frame.
The SkyFrame is always the current frame.

Constructors must set the following properties of the contained ast::FrameSet:
- Include the frames listed above, with specified domains.
    Note that AST will provide the IWC frame when reading standard WCS FITS
    data using an ast::FitsChan if you set "IWC=1" as a property of ast::FitsChan.
    Other constructors may have to work harder.
- For the output ast::SkyFrame set SkyRef=CRVAL (in radians) and SkyRefIs="Ignored".
    This makes CRVAL easy to look up, and "Ignored" prevents CRVAL from being used as an offset.
*/
class SkyWcs : public Transform<Point2Endpoint, SpherePointEndpoint> {
public:
    SkyWcs(SkyWcs const &) = delete;
    /// Move constructor
    SkyWcs(SkyWcs &&) = default;
    SkyWcs &operator=(SkyWcs const &) = delete;
    /// Move assignment operator
    SkyWcs &operator=(SkyWcs &&) = default;

    /**
    Construct a pure tangent SkyWcs

    @param[in] crpix  Center of projection on the CCD using the LSST convention:
                        0, 0 is the lower left pixel of the parent image
    @param[in] crval  Center of projection on the sky
    @param[in] cdMatrix  CD matrix, where element (i-1, j-1) corresponds to FITS keyword CDi_j
                        and i, j have range [1, 2]
    */
    explicit SkyWcs(Point2D const &crpix, SpherePoint const &crval, Eigen::Matrix2d const &cdMatrix);

    /**
    Construct a WCS from FITS keywords

    In addition to standard FITS WCS keywords uses these LSST-specific keywords:
    LTV1, LTV2: offset of subimage with respect to parent image

    TODO implement this using readFrameSet and adding full standardization

    Currently does not strip the used keywords, but that capability should probably be added
    */
    explicit SkyWcs(daf::base::PropertyList &metadata);

    /**
    Construct a WCS from an ast::FrameSet

    This is primarily intended for unpersistence, but usable for other purposes.
    However, for safety the FrameSet must meet the following requirements:
    - Base frame has domain PIXEL0 or ACTUAL_PIXEL0 and 2 axes
    - Frame GRID exists and has 2 axes
    - Frame IWC exists and has 2 axes
    - Current frame is a SkyFrame
    */
    explicit SkyWcs(ast::FrameSet const & frameSet);

    ~SkyWcs(){};

    /**
    Get the pixel scale at the specified pixel position

    The scale is the square root of the area of the specified pixel on the sky.
    */
    Angle getPixelScale(Point2D const &pixel) const;

    /**
    Get CRPIX, the pixel origin, using the LSST convention
    */
    Point2D getPixelOrigin() const;

    /**
    Get CRVAL, the sky origin or celestial fiducial point
    */
    SpherePoint getSkyOrigin() const;

    /**
    Get the CD matrix at the pixel origin
    */
    Eigen::Matrix2d getCdMatrix() const;

    /**
    Get a local TAN WCS approximation to this WCS at the specified pixel position
    */
    SkyWcs getTanWcs(Point2D const &pixel) const;

    /**
    Return a copy of this SkyWcs with the pixel origin by the specified amount
    */
    SkyWcs shiftedPixelOrigin(double dx, double dy) const;

    /**
    Compute sky position(s) from pixel position(s)

    This is another name for tranForward, plus an overload that takes a pair of doubles.
    */
    //@{
    std::pair<Angle, Angle> pixelToSky(double x, double y) const;
    SpherePoint pixelToSky(Point2D const & pixel) const { return tranForward(pixel); };
    std::vector<SpherePoint> pixelToSky(std::vector<Point2D> const & pixels) const {
        return tranForward(pixels);
    }
    //@}

    /**
    Compute pixel position(s) from sky position(s)

    This is another name for tranForward, plus an overload that takes a pair of lsst:afw::geom::Angle.
    */
    //@{
    std::pair<double, double> skyToPixel(Angle const & ra, Angle const & dec) const;
    Point2D skyToPixel(SpherePoint const & sky) const { return tranInverse(sky); }
    std::vector<Point2D> skyToPixel(std::vector<SpherePoint> const & sky) const {
        return tranInverse(sky);
    }
    //@}

protected:
    // Construct a SkyWcs from a shared pointer to an ast::FrameSet
    explicit SkyWcs(std::shared_ptr<ast::FrameSet> &&frameSet);

    // Check a FrameSet to see if it can safely be used for a SkyWcs
    // Return a copy so that it can be used as an argument to the SkyWcs(shared_ptr<FrameSet>) constructor
    std::shared_ptr<ast::FrameSet> _checkFrameSet(ast::FrameSet const & frameSet) const;
};

}  // namespace geom
}  // namespace afw
}  // namespace lsst

#endif
