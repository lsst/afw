// -*- LSST-C++ -*- // fixed format comment for emacs

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

/**
 * \file
 *
 * \ingroup afw
 *
 * \brief Support for warping an %image to a new Wcs.
 */

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <ctime>

#include <memory>
#include "boost/pointer_cast.hpp"
#include "boost/regex.hpp"

#include "lsst/pex/logging/Trace.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/warpExposure.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/gpu/IsGpuBuild.h"
#include "lsst/afw/gpu/GpuExceptions.h"
#include "lsst/afw/gpu/DevicePreference.h"
#include "lsst/afw/math/detail/CudaLanczosWrapper.h"
#include "lsst/afw/math/detail/PositionFunctor.h"
#include "lsst/afw/math/detail/WarpAtOnePoint.h"

namespace pexExcept = lsst::pex::exceptions;
namespace pexLog = lsst::pex::logging;
namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;
namespace afwCoord = lsst::afw::coord;
namespace afwMath = lsst::afw::math;


//
// A helper function for the warping kernels which provides error-checking: 
// the warping kernels are designed to work in two cases
//    0 < x < 1  and ctrX=(size-1)/2
//    -1 < x < 0  and ctrX=(size+1)/2
// (and analogously for y).  Note that to get the second case, Kernel::setCtrX(1) must be
// called before calling Kernel::setKernelParameter().  [see afw::math::offsetImage() for
// an example]
//
// FIXME eventually the 3 warping kernels will inherit from a common base class WarpingKernel
// and this routine can be eliminated by putting the code in WarpingKernel::setKernelParameter()
//
static inline void checkWarpingKernelParameter(const afwMath::SeparableKernel *p, unsigned int ind, double value)
{
    if (ind > 1) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, "bad ind argument in WarpingKernel::setKernelParameter()");
    }
    int ctr = p->getCtr()[ind];
    int size = p->getDimensions()[ind];

    if (ctr == (size-1)/2) {
        if (value < -1e-6 || value > 1+1e-6) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterError, "bad coordinate in WarpingKernel::setKernelParameter()");
        }
    } else if (ctr == (size+1)/2) {
        if (value < -1-1e-6 || value > 1e-6) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterError, "bad coordinate in WarpingKernel::setKernelParameter()");
        }
    } else {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, "bad ctr value in WarpingKernel::setKernelParameter()");
    }
}


PTR(afwMath::Kernel) afwMath::LanczosWarpingKernel::clone() const {
    return PTR(afwMath::Kernel)(new afwMath::LanczosWarpingKernel(this->getOrder()));
}

/**
* @brief get the order of the kernel
*/
int afwMath::LanczosWarpingKernel::getOrder() const {
    return this->getWidth() / 2;
}

void afwMath::LanczosWarpingKernel::setKernelParameter(unsigned int ind, double value) const
{
    checkWarpingKernelParameter(this, ind, value);
    SeparableKernel::setKernelParameter(ind, value);
}

PTR(afwMath::Kernel) afwMath::BilinearWarpingKernel::clone() const {
    return PTR(afwMath::Kernel)(new afwMath::BilinearWarpingKernel());
}

/**
 * \brief Solve bilinear equation
 *
 * Only the following arguments will give reliably meaningful values:
 * *  0.0 or 1.0 if the kernel center index is 0 in this axis
 * * -1.0 or 0.0 if the kernel center index is 1 in this axis
 */
afwMath::Kernel::Pixel afwMath::BilinearWarpingKernel::BilinearFunction1::operator() (double x) const 
{
    //
    // this->_params[0] = value of x where we want to interpolate the function
    // x = integer value of x where we evaluate the function in the interpolation
    // 
    // The following weird-looking expression has no if/else statements, is roundoff-tolerant,
    // and works in the following two cases:
    //     0 < this->_params[0] < 1,  x \in {0,1}
    //     -1 < this->_params[0] < 0,  x \in {-1,0}
    //
    // The checks in BilinearWarpingKernel::setKernelParameter() ensure that one of these
    // conditions is satisfied
    //
    return 0.5 + (1.0 - (2.0 * fabs(this->_params[0]))) * (0.5 - fabs(x));
}

void afwMath::BilinearWarpingKernel::setKernelParameter(unsigned int ind, double value) const
{
    checkWarpingKernelParameter(this, ind, value);
    SeparableKernel::setKernelParameter(ind, value);
}

/**
 * \brief Return string representation.
 */
std::string afwMath::BilinearWarpingKernel::BilinearFunction1::toString(std::string const& prefix) const {
    std::ostringstream os;
    os << "_BilinearFunction1: ";
    os << Function1<Kernel::Pixel>::toString(prefix);
    return os.str();
}

PTR(afwMath::Kernel) afwMath::NearestWarpingKernel::clone() const {
    return PTR(afwMath::Kernel)(new afwMath::NearestWarpingKernel());
}

/**
 * \brief Solve nearest neighbor equation
 *
 * Only the following arguments will give reliably meaningful values:
 * *  0.0 or 1.0 if the kernel center index is 0 in this axis
 * * -1.0 or 0.0 if the kernel center index is 1 in this axis
 */
afwMath::Kernel::Pixel afwMath::NearestWarpingKernel::NearestFunction1::operator() (double x) const {
    // this expression is faster than using conditionals, but offers no sanity checking
    return static_cast<double>((fabs(this->_params[0]) < 0.5) == (fabs(x) < 0.5));
}

void afwMath::NearestWarpingKernel::setKernelParameter(unsigned int ind, double value) const
{
    checkWarpingKernelParameter(this, ind, value);
    SeparableKernel::setKernelParameter(ind, value);
}

/**
 * \brief Return string representation.
 */
std::string afwMath::NearestWarpingKernel::NearestFunction1::toString(std::string const& prefix) const {
    std::ostringstream os;
    os << "_NearestFunction1: ";
    os << Function1<Kernel::Pixel>::toString(prefix);
    return os.str();
}

std::shared_ptr<afwMath::SeparableKernel> afwMath::makeWarpingKernel(std::string name) {
    typedef std::shared_ptr<afwMath::SeparableKernel> KernelPtr;
    boost::cmatch matches;
    static const boost::regex LanczosRE("lanczos(\\d+)");
    if (name == "bilinear") {
        return KernelPtr(new BilinearWarpingKernel());
    } else if (boost::regex_match(name.c_str(), matches, LanczosRE)) {
        std::string orderStr(matches[1].first, matches[1].second);
        int order;
        std::istringstream(orderStr) >> order;
        return KernelPtr(new LanczosWarpingKernel(order));
    } else if (name == "nearest") {
        return KernelPtr(new NearestWarpingKernel());
    } else {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError,
            "unknown warping kernel name: \"" + name + "\"");
    }
}

PTR(afwMath::SeparableKernel) afwMath::WarpingControl::getWarpingKernel() const {
    if (_warpingKernelPtr->getCacheSize() != _cacheSize) {
        _warpingKernelPtr->computeCache(_cacheSize);
    }
    return _warpingKernelPtr;
};

void afwMath::WarpingControl::setWarpingKernelName(
    std::string const &warpingKernelName
) {
    PTR(SeparableKernel) warpingKernelPtr(makeWarpingKernel(warpingKernelName));
    setWarpingKernel(*warpingKernelPtr);
}

void afwMath::WarpingControl::setWarpingKernel(
    SeparableKernel const &warpingKernel
) {
    if (_maskWarpingKernelPtr) {
        _testWarpingKernels(warpingKernel, *_maskWarpingKernelPtr);
    }
    PTR(SeparableKernel) warpingKernelPtr(std::static_pointer_cast<SeparableKernel>(warpingKernel.clone()));
    _testDevicePreference(_devicePreference, warpingKernelPtr);
    _warpingKernelPtr = warpingKernelPtr;
}


PTR(afwMath::SeparableKernel) afwMath::WarpingControl::getMaskWarpingKernel() const {
    if (_maskWarpingKernelPtr) { // lazily update kernel cache
        if (_maskWarpingKernelPtr->getCacheSize() != _cacheSize) {
            _maskWarpingKernelPtr->computeCache(_cacheSize);
        }
    }
    return _maskWarpingKernelPtr;
}

void afwMath::WarpingControl::setMaskWarpingKernelName(
    std::string const &maskWarpingKernelName
) {
    if (!maskWarpingKernelName.empty()) {
        PTR(SeparableKernel) maskWarpingKernelPtr(makeWarpingKernel(maskWarpingKernelName));
        setMaskWarpingKernel(*maskWarpingKernelPtr);
    } else {
        _maskWarpingKernelPtr.reset();
    }
}

void afwMath::WarpingControl::setMaskWarpingKernel(
    SeparableKernel const & maskWarpingKernel
) {
    _testWarpingKernels(*_warpingKernelPtr, maskWarpingKernel);
    _maskWarpingKernelPtr = std::static_pointer_cast<SeparableKernel>(maskWarpingKernel.clone());
}


void afwMath::WarpingControl::_testWarpingKernels(
    SeparableKernel const &warpingKernel,
    SeparableKernel const &maskWarpingKernel
) const {
    lsst::afw::geom::Box2I kernelBBox = lsst::afw::geom::Box2I(
        lsst::afw::geom::Point2I(0, 0) - lsst::afw::geom::Extent2I(warpingKernel.getCtr()),
        warpingKernel.getDimensions()
    );
    lsst::afw::geom::Box2I maskKernelBBox = lsst::afw::geom::Box2I(
        lsst::afw::geom::Point2I(0, 0) - lsst::afw::geom::Extent2I(maskWarpingKernel.getCtr()),
        maskWarpingKernel.getDimensions()
    );
    if (!kernelBBox.contains(maskKernelBBox)) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
            "warping kernel is smaller than mask warping kernel");
    }
}

void afwMath::WarpingControl::_testDevicePreference(
    lsst::afw::gpu::DevicePreference const &devicePreference,
    CONST_PTR(SeparableKernel) const &warpingKernelPtr
) const {
    CONST_PTR(LanczosWarpingKernel) const lanczosKernelPtr =
        std::dynamic_pointer_cast<const LanczosWarpingKernel>(warpingKernelPtr);
    if (devicePreference == lsst::afw::gpu::USE_GPU && !lanczosKernelPtr) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
            "devicePreference = USE_GPU, but warping kernel not Lanczos");
    }
}



template<typename DestExposureT, typename SrcExposureT>
int afwMath::warpExposure(
    DestExposureT &destExposure,       
    SrcExposureT const &srcExposure,  
    afwMath::WarpingControl const &control, 
    typename DestExposureT::MaskedImageT::SinglePixel padValue
    )
{
    if (!destExposure.hasWcs()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, "destExposure has no Wcs");
    }
    if (!srcExposure.hasWcs()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, "srcExposure has no Wcs");
    }
    typename DestExposureT::MaskedImageT mi = destExposure.getMaskedImage();
    std::shared_ptr<afwImage::Calib> calibCopy(new afwImage::Calib(*srcExposure.getCalib()));
    destExposure.setCalib(calibCopy);
    destExposure.setFilter(srcExposure.getFilter());
    return warpImage(mi, *destExposure.getWcs(), srcExposure.getMaskedImage(), *srcExposure.getWcs(),
        control, padValue);
}


/************************************************************************************************************/
namespace {

    inline afwGeom::Point2D computeSrcPos(
            int destCol,  ///< destination column index
            int destRow,  ///< destination row index
            afwGeom::Point2D const &destXY0,    ///< xy0 of destination image
            afwImage::Wcs const &destWcs,       ///< WCS of remapped %image
            afwImage::Wcs const &srcWcs)        ///< WCS of source %image
    {
        double const col = afwImage::indexToPosition(destCol + destXY0[0]);
        double const row = afwImage::indexToPosition(destRow + destXY0[1]);
        afwGeom::Angle sky1, sky2;
        destWcs.pixelToSky(col, row, sky1, sky2);
        return srcWcs.skyToPixel(sky1, sky2);
    }


    inline double computeRelativeArea(
            afwGeom::Point2D const &srcPos,     /// source position at desired destination pixel
            afwGeom::Point2D const &leftSrcPos, /// source position one destination pixel to the left
            afwGeom::Point2D const &upSrcPos)   /// source position one destination pixel above
    {
        afwGeom::Extent2D dSrcA = srcPos - leftSrcPos;
        afwGeom::Extent2D dSrcB = srcPos - upSrcPos;

        return std::abs(dSrcA.getX()*dSrcB.getY() - dSrcA.getY()*dSrcB.getX());
    }

    template<typename DestImageT, typename SrcImageT>
    int doWarpImage(
        DestImageT &destImage,                      ///< remapped %image
        SrcImageT const &srcImage,                  ///< source %image
        afwMath::detail::PositionFunctor const &computeSrcPos,   ///< Functor to compute source position
            ///< called with dest row, column; returns source position (as a Point2D)
        afwMath::WarpingControl const &control,     ///< warping parameters
        typename DestImageT::SinglePixel padValue   ///< value to use for undefined pixels
    ) {
        if (afwMath::details::isSameObject(destImage, srcImage)) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                "destImage is srcImage; cannot warp in place");
        }
        if (destImage.getBBox(afwImage::LOCAL).isEmpty()) {
            return 0;
        }
        // if src image is too small then don't try to warp
        try {
            PTR(afwMath::SeparableKernel) warpingKernelPtr = control.getWarpingKernel();
            warpingKernelPtr->shrinkBBox(srcImage.getBBox(afwImage::LOCAL));
        } catch(...) {
            for (int y = 0, height = destImage.getHeight(); y < height; ++y) {
                for (typename DestImageT::x_iterator destPtr = destImage.row_begin(y), end = destImage.row_end(y);
                    destPtr != end; ++destPtr) {
                    *destPtr = padValue;
                }
            }
            return 0;
        }
        PTR(afwMath::SeparableKernel) warpingKernelPtr = control.getWarpingKernel();
        int interpLength = control.getInterpLength();
        lsst::afw::gpu::DevicePreference devPref = control.getDevicePreference();

        std::shared_ptr<afwMath::LanczosWarpingKernel const> const lanczosKernelPtr =
            std::dynamic_pointer_cast<afwMath::LanczosWarpingKernel>(warpingKernelPtr);

        if (lsst::afw::gpu::isGpuEnabled()) {
            if(!lanczosKernelPtr) {
                if (devPref == lsst::afw::gpu::USE_GPU) {
                    throw LSST_EXCEPT(pexExcept::InvalidParameterError, "Gpu can process only Lanczos kernels");
                }
            } else if (devPref == lsst::afw::gpu::USE_GPU || (lsst::afw::gpu::isGpuBuild() && interpLength > 0) ) {
                PTR(afwMath::SeparableKernel) maskWarpingKernelPtr = control.getWarpingKernel();
                if (control.getMaskWarpingKernel() )
                     maskWarpingKernelPtr = control.getMaskWarpingKernel();
                if (devPref == lsst::afw::gpu::AUTO_WITH_CPU_FALLBACK) {
                    try {
                        std::pair<int, afwMath::detail::WarpImageGpuStatus::ReturnCode> result =
                                           afwMath::detail::warpImageGPU(destImage, srcImage, 
                                                             *lanczosKernelPtr, *maskWarpingKernelPtr, 
                                                             computeSrcPos,  interpLength, padValue, false);
                        if (result.second == afwMath::detail::WarpImageGpuStatus::OK) return result.first;
                    }
                    catch(lsst::afw::gpu::GpuMemoryError) { }
                    catch(pexExcept::MemoryError) { }
                    catch(lsst::afw::gpu::GpuRuntimeError) { }
                } else if (devPref != lsst::afw::gpu::USE_CPU) {
                    std::pair<int, afwMath::detail::WarpImageGpuStatus::ReturnCode> result =
                                           afwMath::detail::warpImageGPU(destImage, srcImage, 
                                                                      *lanczosKernelPtr, *maskWarpingKernelPtr,
                                                                      computeSrcPos, interpLength, padValue,
                                                                      devPref == lsst::afw::gpu::USE_GPU);
                    if (result.second == afwMath::detail::WarpImageGpuStatus::OK) return result.first;
                    if (devPref == lsst::afw::gpu::USE_GPU) {
                        throw LSST_EXCEPT(pexExcept::RuntimeError,
                                          "Gpu cannot perform this warp (kernel too big?)");
                    }
                }
            }
        }
        
        int numGoodPixels = 0;

        // Get the source MaskedImage and a pixel accessor to it.
        int const srcWidth = srcImage.getWidth();
        int const srcHeight = srcImage.getHeight();
        pexLog::TTrace<3>("lsst.afw.math.warp", "source image width=%d; height=%d", srcWidth, srcHeight);

        int const destWidth = destImage.getWidth();
        int const destHeight = destImage.getHeight();

        pexLog::TTrace<3>("lsst.afw.math.warp", "remap image width=%d; height=%d", destWidth, destHeight);

        // Set each pixel of destExposure's MaskedImage
        pexLog::TTrace<4>("lsst.afw.math.warp", "Remapping masked image");

        // A cache of pixel positions on the source corresponding to the previous or current row
        // of the destination image.
        // The first value is for column -1 because the previous source position is used to compute relative area
        // To simplify the indexing, use an iterator that starts at begin+1, thus:
        // srcPosView = _srcPosList.begin() + 1
        // srcPosView[col-1] and lower indices are for this row
        // srcPosView[col] and higher indices are for the previous row
        std::vector<afwGeom::Point2D> _srcPosList(1 + destWidth);
        std::vector<afwGeom::Point2D>::iterator const srcPosView = _srcPosList.begin() + 1;

        int const maxCol = destWidth - 1;
        int const maxRow = destHeight - 1;
        
        afwMath::detail::WarpAtOnePoint<DestImageT, SrcImageT> warpAtOnePoint(srcImage, control, padValue);

        if (interpLength > 0) {
            // Use interpolation. Note that 1 produces the same result as no interpolation
            // but uses this code branch, thus providing an easy way to compare the two branches.

            // Estimate for number of horizontal interpolation band edges, to reserve memory in vectors
            int const numColEdges = 2 + ((destWidth - 1) / interpLength);

            // A list of edge column indices for interpolation bands;
            // starts at -1, increments by interpLen (except the final interval), and ends at destWidth-1
            std::vector<int> edgeColList;
            edgeColList.reserve(numColEdges);

            // A list of 1/column width for horizontal interpolation bands; the first value is garbage.
            // The inverse is used for speed because the values is always multiplied.
            std::vector<double> invWidthList;
            invWidthList.reserve(numColEdges);

            // Compute edgeColList and invWidthList
            edgeColList.push_back(-1);
            invWidthList.push_back(0.0);
            for (int prevEndCol = -1; prevEndCol < maxCol; prevEndCol += interpLength) {
                int endCol = prevEndCol + interpLength;
                if (endCol > maxCol) {
                    endCol = maxCol;
                }
                edgeColList.push_back(endCol);
                assert(endCol - prevEndCol > 0);
                invWidthList.push_back(1.0 / static_cast<double>(endCol - prevEndCol));
            }
            assert(edgeColList.back() == maxCol);

            // A list of delta source positions along the edge columns of the horizontal interpolation bands
            std::vector<afwGeom::Extent2D> yDeltaSrcPosList(edgeColList.size());

            // Initialize _srcPosList for row -1
            //srcPosView[-1] = computeSrcPos(-1, -1, destXY0, destWcs, srcWcs);
            srcPosView[-1] = computeSrcPos(-1, -1);
            for (int colBand = 1, endBand = edgeColList.size(); colBand < endBand; ++colBand) {
                int const prevEndCol = edgeColList[colBand-1];
                int const endCol = edgeColList[colBand];
                afwGeom::Point2D leftSrcPos = srcPosView[prevEndCol];
                afwGeom::Point2D rightSrcPos = computeSrcPos(endCol, -1);
                afwGeom::Extent2D xDeltaSrcPos = (rightSrcPos - leftSrcPos) * invWidthList[colBand];

                for (int col = prevEndCol + 1; col <= endCol; ++col) {
                    srcPosView[col] = srcPosView[col-1] + xDeltaSrcPos;
                }
            }

            int endRow = -1;
            while (endRow < maxRow) {
                // Next horizontal interpolation band

                int prevEndRow = endRow;
                endRow = prevEndRow + interpLength;
                if (endRow > maxRow) {
                    endRow = maxRow;
                }
                assert(endRow - prevEndRow > 0);
                double interpInvHeight = 1.0 / static_cast<double>(endRow - prevEndRow);

                // Set yDeltaSrcPosList for this horizontal interpolation band
                for (int colBand = 0, endBand = edgeColList.size(); colBand < endBand; ++colBand) {
                    int endCol = edgeColList[colBand];
                    afwGeom::Point2D bottomSrcPos = computeSrcPos(endCol, endRow);
                    yDeltaSrcPosList[colBand] = (bottomSrcPos - srcPosView[endCol]) * interpInvHeight;
                }

                for (int row = prevEndRow + 1; row <= endRow; ++row) {
                    typename DestImageT::x_iterator destXIter = destImage.row_begin(row);
                    srcPosView[-1] += yDeltaSrcPosList[0];
                    for (int colBand = 1, endBand = edgeColList.size(); colBand < endBand; ++colBand) {
                        /// Next vertical interpolation band

                        int const prevEndCol = edgeColList[colBand-1];
                        int const endCol = edgeColList[colBand];

                        // Compute xDeltaSrcPos; remember that srcPosView contains
                        // positions for this row in prevEndCol and smaller indices,
                        // and positions for the previous row for larger indices (including endCol)
                        afwGeom::Point2D leftSrcPos = srcPosView[prevEndCol];
                        afwGeom::Point2D rightSrcPos = srcPosView[endCol] + yDeltaSrcPosList[colBand];
                        afwGeom::Extent2D xDeltaSrcPos = (rightSrcPos - leftSrcPos) * invWidthList[colBand];

                        for (int col = prevEndCol + 1; col <= endCol; ++col, ++destXIter) {
                            afwGeom::Point2D leftSrcPos = srcPosView[col-1];
                            afwGeom::Point2D srcPos = leftSrcPos + xDeltaSrcPos;
                            double relativeArea = computeRelativeArea(srcPos, leftSrcPos, srcPosView[col]);

                            srcPosView[col] = srcPos;

                            if (warpAtOnePoint(destXIter, srcPos, relativeArea,
                                typename lsst::afw::image::detail::image_traits<DestImageT>::image_category())) {
                                ++numGoodPixels;
                            }
                        } // for col
                    }   // for col band
                }   // for row
            }   // while next row band


        } else {
            // No interpolation

            // initialize _srcPosList for row -1;
            // the first value is not needed, but it's safer to compute it
            std::vector<afwGeom::Point2D>::iterator srcPosView = _srcPosList.begin() + 1;
            for (int col = -1; col < destWidth; ++col) {
                srcPosView[col] = computeSrcPos(col, -1);
            }

            for (int row = 0; row < destHeight; ++row) {
                typename DestImageT::x_iterator destXIter = destImage.row_begin(row);

                srcPosView[-1] = computeSrcPos(-1, row);

                for (int col = 0; col < destWidth; ++col, ++destXIter) {
                    afwGeom::Point2D srcPos = computeSrcPos(col, row);
                    double relativeArea = computeRelativeArea(srcPos, srcPosView[col-1], srcPosView[col]);
                    srcPosView[col] = srcPos;
                    
                    if (warpAtOnePoint(destXIter, srcPos, relativeArea,
                        typename lsst::afw::image::detail::image_traits<DestImageT>::image_category())) {
                        ++numGoodPixels;
                    }
                }   // for col
            }   // for row
        } // if interp

        return numGoodPixels;
    }

} // namespace

template<typename DestImageT, typename SrcImageT>
int afwMath::warpImage(
    DestImageT &destImage,
    lsst::afw::image::Wcs const &destWcs,
    SrcImageT const &srcImage,
    lsst::afw::image::Wcs const &srcWcs,
    afwMath::WarpingControl const &control,
    typename DestImageT::SinglePixel padValue
) {
    afwGeom::Point2D const destXY0(destImage.getXY0());
    afwImage::XYTransformFromWcsPair xyTransform{destWcs.clone(), srcWcs.clone()};
    afwMath::detail::XYTransformPositionFunctor const computeSrcPos{destXY0, xyTransform};
    return doWarpImage(destImage, srcImage, computeSrcPos, control, padValue);
}


template<typename DestImageT, typename SrcImageT>
int afwMath::warpImage(
    DestImageT &destImage,
    SrcImageT const &srcImage,
    afwGeom::XYTransform const &xyTransform,
    afwMath::WarpingControl const &control,
    typename DestImageT::SinglePixel padValue
) {
    afwGeom::Point2D const destXY0(destImage.getXY0());
    afwMath::detail::XYTransformPositionFunctor const computeSrcPos(destXY0, xyTransform);
    return doWarpImage(destImage, srcImage, computeSrcPos, control, padValue);
}


template<typename DestImageT, typename SrcImageT>
int afwMath::warpCenteredImage(
    DestImageT &destImage,
    SrcImageT const &srcImage,
    afwGeom::LinearTransform const &linearTransform,
    afwGeom::Point2D const &centerPosition,
    afwMath::WarpingControl const &control,
    typename DestImageT::SinglePixel padValue
) {
    // force src and dest to be the same size and xy0
    if (
        (destImage.getWidth() != srcImage.getWidth()) ||
        (destImage.getHeight() != srcImage.getHeight()) ||
        (destImage.getXY0() != srcImage.getXY0())
       ) {
        std::ostringstream errStream;
        errStream << "src and dest images must have same size and xy0.";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, errStream.str());
    }

    // set the xy0 coords to 0,0 to make life easier
    SrcImageT srcImageCopy(srcImage, true);
    srcImageCopy.setXY0(0, 0);
    destImage.setXY0(0, 0);
    afwGeom::Extent2D cLocal = afwGeom::Extent2D(centerPosition) - afwGeom::Extent2D(srcImage.getXY0());

    // for the affine transform, the centerPosition will not only get sheared, but also
    // moved slightly.  So we'll include a translation to move it back by an amount
    // centerPosition - translatedCenterPosition
    afwGeom::AffineTransform affTran(linearTransform, cLocal - linearTransform(cLocal));
    afwGeom::AffineXYTransform affXYTransform(affTran);

    // now warp
#if 0
    static float t = 0.0;
    float t_before = 1.0*clock()/CLOCKS_PER_SEC;
    int n = warpImage(destImage, srcImageCopy, affTran, control, padValue);
    float t_after = 1.0*clock()/CLOCKS_PER_SEC;
    float dt = t_after - t_before;
    t += dt;
    std::cout <<srcImage.getWidth()<<"x"<<srcImage.getHeight()<<": "<< dt <<" "<< t <<std::endl;
#else
    int n = warpImage(destImage, srcImageCopy, affXYTransform, control, padValue);
#endif

    // fix the origin and we're done.
    destImage.setXY0(srcImage.getXY0());

    return n;
}


//
// Explicit instantiations
//
/// \cond
// may need to omit default params for EXPOSURE -- original code did that and it worked
#define EXPOSURE(PIXTYPE) afwImage::Exposure<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
#define MASKEDIMAGE(PIXTYPE) afwImage::MaskedImage<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
#define IMAGE(PIXTYPE) afwImage::Image<PIXTYPE>
#define NL /* */

#define INSTANTIATE(DESTIMAGEPIXELT, SRCIMAGEPIXELT) \
    template int afwMath::warpCenteredImage( \
        IMAGE(DESTIMAGEPIXELT) &destImage, \
        IMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwGeom::LinearTransform const &linearTransform, \
        afwGeom::Point2D const &centerPosition, \
        afwMath::WarpingControl const &control, \
        IMAGE(DESTIMAGEPIXELT)::SinglePixel padValue); NL \
    template int afwMath::warpCenteredImage( \
        MASKEDIMAGE(DESTIMAGEPIXELT) &destImage, \
        MASKEDIMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwGeom::LinearTransform const &linearTransform, \
        afwGeom::Point2D const &centerPosition, \
        afwMath::WarpingControl const &control, \
        MASKEDIMAGE(DESTIMAGEPIXELT)::SinglePixel padValue); NL \
    template int afwMath::warpImage( \
        IMAGE(DESTIMAGEPIXELT) &destImage, \
        IMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwGeom::XYTransform const &xyTransform, \
        afwMath::WarpingControl const &control, \
        IMAGE(DESTIMAGEPIXELT)::SinglePixel padValue); NL \
    template int afwMath::warpImage( \
        MASKEDIMAGE(DESTIMAGEPIXELT) &destImage, \
        MASKEDIMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwGeom::XYTransform const &xyTransform, \
        afwMath::WarpingControl const &control, \
        MASKEDIMAGE(DESTIMAGEPIXELT)::SinglePixel padValue); NL \
    template int afwMath::warpImage( \
        IMAGE(DESTIMAGEPIXELT) &destImage, \
        afwImage::Wcs const &destWcs, \
        IMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwImage::Wcs const &srcWcs, \
        afwMath::WarpingControl const &control, \
        IMAGE(DESTIMAGEPIXELT)::SinglePixel padValue); NL \
    template int afwMath::warpImage( \
        MASKEDIMAGE(DESTIMAGEPIXELT) &destImage, \
        afwImage::Wcs const &destWcs, \
        MASKEDIMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwImage::Wcs const &srcWcs, \
        afwMath::WarpingControl const &control, \
        MASKEDIMAGE(DESTIMAGEPIXELT)::SinglePixel padValue); NL \
    template int afwMath::warpExposure( \
        EXPOSURE(DESTIMAGEPIXELT) &destExposure, \
        EXPOSURE(SRCIMAGEPIXELT) const &srcExposure, \
        afwMath::WarpingControl const &control,\
        EXPOSURE(DESTIMAGEPIXELT)::MaskedImageT::SinglePixel padValue);




INSTANTIATE(double, double)
INSTANTIATE(double, float)
INSTANTIATE(double, int)
INSTANTIATE(double, std::uint16_t)
INSTANTIATE(float, float)
INSTANTIATE(float, int)
INSTANTIATE(float, std::uint16_t)
INSTANTIATE(int, int)
INSTANTIATE(std::uint16_t, std::uint16_t)
/// \endcond
