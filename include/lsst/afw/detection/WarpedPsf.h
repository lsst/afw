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

#include "lsst/daf/base.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/XYTransform.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/Color.h"
#include "lsst/afw/detection/Psf.h"

#ifndef LSST_AFW_DETECTION_WARPEDPSF_H
#define LSST_AFW_DETECTION_WARPEDPSF_H

namespace lsst {
namespace afw {
namespace detection {

/**
 * \file
 * @brief WarpedPsf: a class which combines an unwarped psf and a camera distortion
 *
 * If K_0(x,x') is the unwarped PSF, and f is the camera distortion, then the 
 * warped PSF is defined by
 *
 *   K(f(x),f(x')) = K_0(x,x')      (*)
 *
 * We linearize the camera distortion in the vicinity of the point where the
 * PSF is computed.  The definition (*) does not include the Jacobian of the
 * transformation, since the afw convention is that PSF's are normalized to
 * have integral 1 anyway.
 *
 * Note: In order to plug into a WarpedPsf, the undistorted Psf only needs to define
 * the virtuals clone() and doGetLocalKernel().
 */
class WarpedPsf : public Psf {
public:
    typedef afw::image::Color Color;
    typedef afw::math::Kernel Kernel;
    typedef afw::geom::Point2I Point2I;
    typedef afw::geom::Point2D Point2D;
    typedef afw::geom::Extent2I Extent2I;
    typedef afw::geom::XYTransform XYTransform;

    /**
     * @brief Construct WarpedPsf from unwarped psf and distortion.
     *
     * If p is the nominal pixel position, and p' is the true position on the sky, then our
     * convention for the transform is that p' = distortion.forwardTransform(p)
     */
    WarpedPsf(CONST_PTR(Psf) undistorted_psf, CONST_PTR(XYTransform) distortion);

protected:

    virtual PTR(Psf) clone() const;

    virtual PTR(Image) doComputeKernelImage(geom::Point2D const & ccdXY, image::Color const & color) const;

protected:
    CONST_PTR(Psf) _undistorted_psf;
    CONST_PTR(XYTransform) _distortion;

    /**
     * @brief This helper function is used in doComputeImage() and doGetLocalKernel().
     *
     * The image returned by this routine is a "kernel image", i.e. xy0 is not meaningful
     * but there is a distinguished central pixel which corresponds to the point "p" where
     * the PSF is evaluated.
     */
    PTR(Image) _makeWarpedKernelImage(Point2D const &p, Color const &color, Point2I &ctr) const;
};


}}}

#endif  // LSST_AFW_DETECTION_WARPEDPSF_H
