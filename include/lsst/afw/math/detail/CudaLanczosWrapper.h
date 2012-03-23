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

    class SrcPosFunctor {
    public:
        SrcPosFunctor() {}
        typedef boost::shared_ptr<SrcPosFunctor> Ptr;
        virtual lsst::afw::geom::Point2D operator()(int destCol, int destRow) const = 0;
    private:
    };

    class WcsSrcPosFunctor : public SrcPosFunctor {
    public:
        WcsSrcPosFunctor(
                         lsst::afw::geom::Point2D const &destXY0,    ///< xy0 of destination image
                         lsst::afw::image::Wcs const &destWcs,       ///< WCS of remapped %image
                         lsst::afw::image::Wcs const &srcWcs
                        ) :      ///< WCS of source %image
            SrcPosFunctor(),
            _destXY0(destXY0),
            _destWcs(destWcs),
            _srcWcs(srcWcs) {}
        typedef boost::shared_ptr<WcsSrcPosFunctor> Ptr;

        virtual lsst::afw::geom::Point2D operator()(int destCol, int destRow) const {
            double const col = lsst::afw::image::indexToPosition(destCol + _destXY0[0]);
            double const row = lsst::afw::image::indexToPosition(destRow + _destXY0[1]);
            lsst::afw::geom::Angle sky1, sky2;
            _destWcs.pixelToSky(col, row, sky1, sky2);
            return _srcWcs.skyToPixel(sky1, sky2);
        }
    private:
        lsst::afw::geom::Point2D const &_destXY0;
        lsst::afw::image::Wcs const &_destWcs;
        lsst::afw::image::Wcs const &_srcWcs;
    };

/**
 * \brief GPU accelerated image warping for Lanczos resampling
 *
 * \return a std::pair<int,bool> containing:
 *                1) the number of valid pixels in destImage (those that are not edge pixels).
 *                2) whether the warping was performed (if false, then the first value is not defined)
 *
 * This function requires a Lanczos warping kernel to perform the source value estimation.
 *
 * This function will not perform the warping if kernel size is too large.
 * (currently, when the order of the Lanczos kernel is >50)
 * If warping is not performed, the return value will be (X,false).
 * If forceProcessing is true:
 *       - this function will throw exceptions if a GPU device cannot be selected or used
 * If forceProcessing is false:
 *       - the warping will not be performed if the GPU code path is estimated to be slower than CPU code path.
 *              That might happen if interpLength is too small (less than 3)
 *       - the warping will not be performed if a GPU device cannot be selected or used
 *
 * Also see lsst::afw::math::warpImage()
 *
 * \b Implementation:
 * Calculates samples of the coordinate transform function at some points, which are spaced by interpLength intervals
 * Calls CalculateInterpolationData() for coordinate transformation function samples.
 * Calls WarpImageGpuWrapper() to perform the wapring.
 *
 * \throw lsst::pex::exceptions::InvalidParameterException if interpLength < 1
 * \throw lsst::pex::exceptions::MemoryException when allocation of CPU memory fails
 * \throw lsst::afw::gpu::GpuMemoryException when allocation or transfer to/from GPU memory fails
 * \throw lsst::afw::gpu::GpuRuntimeErrorException when GPU code run fails
 *
 */
template<typename DestImageT, typename SrcImageT>
std::pair<int,bool> warpImageGPU(
    DestImageT &destImage,                  ///< remapped %image
    SrcImageT const &srcImage,              ///< source %image
    lsst::afw::math::LanczosWarpingKernel const &warpingKernel,   ///< warping kernel
    SrcPosFunctor const &computeSrcPos,      ///< Functor to compute source position
    int const interpLength,                  ///< Distance over which WCS can be linearily interpolated
                                             ///< must be >0
    typename DestImageT::SinglePixel padValue, ///< value to use for undefined pixels
    const bool forceProcessing=true          ///< if true, this function will perform the warping even when
                                             ///< it is slower then the CPU code path
    );

}}}} //namespace lsst::afw::math::detail ends

