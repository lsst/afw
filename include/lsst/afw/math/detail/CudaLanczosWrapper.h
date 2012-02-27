// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * Copyright 2008 - 2012 LSST Corporation.
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

/**
 * @file
 *
 * @brief GPU accelerared image warping
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

namespace lsst {
namespace afw {
namespace math {
namespace detail {

    inline lsst::afw::geom::Point2D computeSrcPos(
            int destCol,  ///< destination column index
            int destRow,  ///< destination row index
            lsst::afw::geom::Point2D const &destXY0,    ///< xy0 of destination image
            lsst::afw::image::Wcs const &destWcs,       ///< WCS of remapped %image
            lsst::afw::image::Wcs const &srcWcs)        ///< WCS of source %image
    {
        double const col = lsst::afw::image::indexToPosition(destCol + destXY0[0]);
        double const row = lsst::afw::image::indexToPosition(destRow + destXY0[1]);
        lsst::afw::geom::Angle sky1, sky2;
        destWcs.pixelToSky(col, row, sky1, sky2);
        return srcWcs.skyToPixel(sky1, sky2);
    }

/**
 * \brief GPU accelerated image warping for Lanczos resampling
 *
 * \return a std::pair<int,bool> containing:
 *                1) the number of valid pixels in destImage (those that are not edge pixels).
 *                2) whether the warping was performed (if false, then the first value is not defined)
 *
 * This function requires a warping kernel to perform the interpolation.
 * Only Lanczos kernel is a valid input, any other kernel will raise in exception
 *
 * This function will not perform the warping if kernel size is too large.
 * (currently, when the order of the Lanczos kernel is >50)
 * If warping is not performed, the return value will be (X,false).
 * If forceProcessing is false, the warping might not be performed if interpLength is too small
 *
 * Also see lsst::afw::math::warpImage()
 *
 * \b Implementation:
 * Calculates values of the coordinate transform function at some points, which are spaced by interpLength intervals
 * Calls CalculateInterpolationData().
 * Calls WarpImageGpuWrapper() to perform the wapring.
 *
 * \throw lsst::pex::exceptions::InvalidParameterException if the warping kernel is not a Lanczos kernel
 * \throw lsst::pex::exceptions::InvalidParameterException if interpLength < 1
 * \throw lsst::pex::exceptions::MemoryException when allocation of CPU memory fails
 * \throw lsst::afw::gpu::GpuMemoryException when allocation or transfer to/from GPU memory fails
 * \throw lsst::afw::gpu::GpuRuntimeErrorException when GPU code run fails
 *
 */
template<typename DestImageT, typename SrcImageT>
std::pair<int,bool> warpImageGPU(
    DestImageT &destImage,                  ///< remapped %image
    lsst::afw::image::Wcs const &destWcs,   ///< WCS of remapped %image
    SrcImageT const &srcImage,              ///< source %image
    lsst::afw::image::Wcs const &srcWcs,               ///< WCS of source %image
    lsst::afw::math::SeparableKernel &warpingKernel,   ///< warping kernel; determines warping algorithm
    int const interpLength,                  ///< Distance over which WCS can be linearily interpolated
                                             ///< must be >0
    lsst::afw::gpu::DevicePreference devPref
    );

}}}} //namespace lsst::afw::math::detail ends

