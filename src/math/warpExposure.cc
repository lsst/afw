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
/*
 * Support for warping an %image to a new Wcs.
 */

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <ctime>

#include <memory>
#include "boost/pointer_cast.hpp"
#include "boost/regex.hpp"
#include "astshim.h"

#include "lsst/log/Log.h"
#include "lsst/pex/exceptions.h"
#include "lsst/geom.h"
#include "lsst/afw/math/warpExposure.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/afw/math/detail/WarpAtOnePoint.h"

namespace pexExcept = lsst::pex::exceptions;

using std::swap;

namespace lsst {
namespace afw {
namespace math {

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
static inline void checkWarpingKernelParameter(const SeparableKernel *p, unsigned int ind, double value) {
    if (ind > 1) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                          "bad ind argument in WarpingKernel::setKernelParameter()");
    }
    int ctr = p->getCtr()[ind];
    int size = p->getDimensions()[ind];

    if (ctr == (size - 1) / 2) {
        if (value < -1e-6 || value > 1 + 1e-6) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                              "bad coordinate in WarpingKernel::setKernelParameter()");
        }
    } else if (ctr == (size + 1) / 2) {
        if (value < -1 - 1e-6 || value > 1e-6) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                              "bad coordinate in WarpingKernel::setKernelParameter()");
        }
    } else {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                          "bad ctr value in WarpingKernel::setKernelParameter()");
    }
}

std::shared_ptr<Kernel> LanczosWarpingKernel::clone() const {
    return std::shared_ptr<Kernel>(new LanczosWarpingKernel(this->getOrder()));
}

int LanczosWarpingKernel::getOrder() const { return this->getWidth() / 2; }

void LanczosWarpingKernel::setKernelParameter(unsigned int ind, double value) const {
    checkWarpingKernelParameter(this, ind, value);
    SeparableKernel::setKernelParameter(ind, value);
}

std::shared_ptr<Kernel> BilinearWarpingKernel::clone() const {
    return std::shared_ptr<Kernel>(new BilinearWarpingKernel());
}

Kernel::Pixel BilinearWarpingKernel::BilinearFunction1::operator()(double x) const {
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

void BilinearWarpingKernel::setKernelParameter(unsigned int ind, double value) const {
    checkWarpingKernelParameter(this, ind, value);
    SeparableKernel::setKernelParameter(ind, value);
}

std::string BilinearWarpingKernel::BilinearFunction1::toString(std::string const &prefix) const {
    std::ostringstream os;
    os << "_BilinearFunction1: ";
    os << Function1<Kernel::Pixel>::toString(prefix);
    return os.str();
}

std::shared_ptr<Kernel> NearestWarpingKernel::clone() const {
    return std::make_shared<NearestWarpingKernel>();
}

Kernel::Pixel NearestWarpingKernel::NearestFunction1::operator()(double x) const {
    // this expression is faster than using conditionals, but offers no sanity checking
    return static_cast<double>((fabs(this->_params[0]) < 0.5) == (fabs(x) < 0.5));
}

void NearestWarpingKernel::setKernelParameter(unsigned int ind, double value) const {
    checkWarpingKernelParameter(this, ind, value);
    SeparableKernel::setKernelParameter(ind, value);
}

std::string NearestWarpingKernel::NearestFunction1::toString(std::string const &prefix) const {
    std::ostringstream os;
    os << "_NearestFunction1: ";
    os << Function1<Kernel::Pixel>::toString(prefix);
    return os.str();
}

std::shared_ptr<SeparableKernel> makeWarpingKernel(std::string name) {
    typedef std::shared_ptr<SeparableKernel> KernelPtr;
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
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, "unknown warping kernel name: \"" + name + "\"");
    }
}

std::shared_ptr<SeparableKernel> WarpingControl::getWarpingKernel() const {
    if (_warpingKernelPtr->getCacheSize() != _cacheSize) {
        _warpingKernelPtr->computeCache(_cacheSize);
    }
    return _warpingKernelPtr;
};

void WarpingControl::setWarpingKernelName(std::string const &warpingKernelName) {
    std::shared_ptr<SeparableKernel> warpingKernelPtr(makeWarpingKernel(warpingKernelName));
    setWarpingKernel(*warpingKernelPtr);
}

void WarpingControl::setWarpingKernel(SeparableKernel const &warpingKernel) {
    if (_maskWarpingKernelPtr) {
        _testWarpingKernels(warpingKernel, *_maskWarpingKernelPtr);
    }
    std::shared_ptr<SeparableKernel> warpingKernelPtr(
            std::static_pointer_cast<SeparableKernel>(warpingKernel.clone()));
    _warpingKernelPtr = warpingKernelPtr;
}

std::shared_ptr<SeparableKernel> WarpingControl::getMaskWarpingKernel() const {
    if (_maskWarpingKernelPtr) {  // lazily update kernel cache
        if (_maskWarpingKernelPtr->getCacheSize() != _cacheSize) {
            _maskWarpingKernelPtr->computeCache(_cacheSize);
        }
    }
    return _maskWarpingKernelPtr;
}

void WarpingControl::setMaskWarpingKernelName(std::string const &maskWarpingKernelName) {
    if (!maskWarpingKernelName.empty()) {
        std::shared_ptr<SeparableKernel> maskWarpingKernelPtr(makeWarpingKernel(maskWarpingKernelName));
        setMaskWarpingKernel(*maskWarpingKernelPtr);
    } else {
        _maskWarpingKernelPtr.reset();
    }
}

void WarpingControl::setMaskWarpingKernel(SeparableKernel const &maskWarpingKernel) {
    _testWarpingKernels(*_warpingKernelPtr, maskWarpingKernel);
    _maskWarpingKernelPtr = std::static_pointer_cast<SeparableKernel>(maskWarpingKernel.clone());
}

void WarpingControl::_testWarpingKernels(SeparableKernel const &warpingKernel,
                                         SeparableKernel const &maskWarpingKernel) const {
    lsst::geom::Box2I kernelBBox = lsst::geom::Box2I(lsst::geom::Point2I(0, 0) - lsst::geom::Extent2I(warpingKernel.getCtr()),
                                         warpingKernel.getDimensions());
    lsst::geom::Box2I maskKernelBBox = lsst::geom::Box2I(lsst::geom::Point2I(0, 0) - lsst::geom::Extent2I(maskWarpingKernel.getCtr()),
                                             maskWarpingKernel.getDimensions());
    if (!kernelBBox.contains(maskKernelBBox)) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                          "warping kernel is smaller than mask warping kernel");
    }
}

template <typename DestExposureT, typename SrcExposureT>
int warpExposure(DestExposureT &destExposure, SrcExposureT const &srcExposure, WarpingControl const &control,
                 typename DestExposureT::MaskedImageT::SinglePixel padValue) {
    if (!destExposure.hasWcs()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, "destExposure has no Wcs");
    }
    if (!srcExposure.hasWcs()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, "srcExposure has no Wcs");
    }
    typename DestExposureT::MaskedImageT mi = destExposure.getMaskedImage();
    std::shared_ptr<image::Calib> calibCopy(new image::Calib(*srcExposure.getCalib()));
    destExposure.setCalib(calibCopy);
    destExposure.setFilter(srcExposure.getFilter());
    destExposure.getInfo()->setVisitInfo(srcExposure.getInfo()->getVisitInfo());
    return warpImage(mi, *destExposure.getWcs(), srcExposure.getMaskedImage(), *srcExposure.getWcs(), control,
                     padValue);
}

namespace {

inline lsst::geom::Point2D computeSrcPos(int destCol,                   ///< @internal destination column index
                                   int destRow,                   ///< @internal destination row index
                                   lsst::geom::Point2D const &destXY0,  ///< @internal xy0 of destination image
                                   geom::SkyWcs const &destWcs,   ///< @internal WCS of remapped %image
                                   geom::SkyWcs const &srcWcs)    ///< @internal WCS of source %image
{
    lsst::geom::Point2D const destPix(image::indexToPosition(destCol + destXY0[0]),
                                image::indexToPosition(destRow + destXY0[1]));
    return srcWcs.skyToPixel(destWcs.pixelToSky(destPix));
}

inline double computeRelativeArea(
        lsst::geom::Point2D const &srcPos,      /// @internal source position at desired destination pixel
        lsst::geom::Point2D const &leftSrcPos,  /// @internal source position one destination pixel to the left
        lsst::geom::Point2D const &upSrcPos)    /// @internal source position one destination pixel above
{
    lsst::geom::Extent2D dSrcA = srcPos - leftSrcPos;
    lsst::geom::Extent2D dSrcB = srcPos - upSrcPos;

    return std::abs(dSrcA.getX() * dSrcB.getY() - dSrcA.getY() * dSrcB.getX());
}

}  // namespace

template <typename DestImageT, typename SrcImageT>
int warpImage(DestImageT &destImage, geom::SkyWcs const &destWcs, SrcImageT const &srcImage,
              geom::SkyWcs const &srcWcs, WarpingControl const &control,
              typename DestImageT::SinglePixel padValue) {
    auto srcToDest = geom::makeWcsPairTransform(srcWcs, destWcs);
    return warpImage(destImage, srcImage, *srcToDest, control, padValue);
}

template <typename DestImageT, typename SrcImageT>
int warpImage(DestImageT &destImage, SrcImageT const &srcImage,
              geom::TransformPoint2ToPoint2 const &srcToDest, WarpingControl const &control,
              typename DestImageT::SinglePixel padValue) {
    if (imagesOverlap(destImage, srcImage)) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, "destImage overlaps srcImage; cannot warp");
    }
    if (destImage.getBBox(image::LOCAL).isEmpty()) {
        return 0;
    }
    // if src image is too small then don't try to warp
    std::shared_ptr<SeparableKernel> warpingKernelPtr = control.getWarpingKernel();
    try {
        warpingKernelPtr->shrinkBBox(srcImage.getBBox(image::LOCAL));
    } catch (lsst::pex::exceptions::InvalidParameterError) {
        for (int y = 0, height = destImage.getHeight(); y < height; ++y) {
            for (typename DestImageT::x_iterator destPtr = destImage.row_begin(y), end = destImage.row_end(y);
                 destPtr != end; ++destPtr) {
                *destPtr = padValue;
            }
        }
        return 0;
    }
    int interpLength = control.getInterpLength();

    std::shared_ptr<LanczosWarpingKernel const> const lanczosKernelPtr =
            std::dynamic_pointer_cast<LanczosWarpingKernel>(warpingKernelPtr);

    int numGoodPixels = 0;

    // compute a transform from local destination pixels to parent source pixels
    auto const parentDestToParentSrc = srcToDest.getInverse();
    std::vector<double> const localDestToParentDestVec = {static_cast<double>(destImage.getX0()),
                                                          static_cast<double>(destImage.getY0())};
    auto const localDestToParentDest = geom::TransformPoint2ToPoint2(ast::ShiftMap(localDestToParentDestVec));
    auto const localDestToParentSrc = localDestToParentDest.then(*parentDestToParentSrc);

    // Get the source MaskedImage and a pixel accessor to it.
    int const srcWidth = srcImage.getWidth();
    int const srcHeight = srcImage.getHeight();
    LOGL_DEBUG("TRACE2.afw.math.warp", "source image width=%d; height=%d", srcWidth, srcHeight);

    int const destWidth = destImage.getWidth();
    int const destHeight = destImage.getHeight();
    LOGL_DEBUG("TRACE2.afw.math.warp", "remap image width=%d; height=%d", destWidth, destHeight);

    // Set each pixel of destExposure's MaskedImage
    LOGL_DEBUG("TRACE3.afw.math.warp", "Remapping masked image");

    int const maxCol = destWidth - 1;
    int const maxRow = destHeight - 1;

    detail::WarpAtOnePoint<DestImageT, SrcImageT> warpAtOnePoint(srcImage, control, padValue);

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
        // The inverse is used for speed because the values are always multiplied.
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
        std::vector<lsst::geom::Extent2D> yDeltaSrcPosList(edgeColList.size());

        // A cache of pixel positions on the source corresponding to the previous or current row
        // of the destination image.
        // The first value is for column -1 because the previous source position is used to compute relative
        // area To simplify the indexing, use an iterator that starts at begin+1, thus: srcPosView =
        // srcPosList.begin() + 1 srcPosView[col-1] and lower indices are for this row srcPosView[col] and
        // higher indices are for the previous row
        std::vector<lsst::geom::Point2D> srcPosList(1 + destWidth);
        std::vector<lsst::geom::Point2D>::iterator const srcPosView = srcPosList.begin() + 1;

        std::vector<lsst::geom::Point2D> endColPosList;
        endColPosList.reserve(numColEdges);

        // Initialize srcPosList for row -1
        for (int colBand = 0, endBand = edgeColList.size(); colBand < endBand; ++colBand) {
            int const endCol = edgeColList[colBand];
            endColPosList.emplace_back(lsst::geom::Point2D(endCol, -1));
        }
        auto rightSrcPosList = localDestToParentSrc->applyForward(endColPosList);
        srcPosView[-1] = rightSrcPosList[0];
        for (int colBand = 1, endBand = edgeColList.size(); colBand < endBand; ++colBand) {
            int const prevEndCol = edgeColList[colBand - 1];
            int const endCol = edgeColList[colBand];
            lsst::geom::Point2D leftSrcPos = srcPosView[prevEndCol];

            lsst::geom::Extent2D xDeltaSrcPos = (rightSrcPosList[colBand] - leftSrcPos) * invWidthList[colBand];

            for (int col = prevEndCol + 1; col <= endCol; ++col) {
                srcPosView[col] = srcPosView[col - 1] + xDeltaSrcPos;
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
            std::vector<lsst::geom::Point2D> destRowPosList;
            destRowPosList.reserve(edgeColList.size());
            for (int colBand = 0, endBand = edgeColList.size(); colBand < endBand; ++colBand) {
                int endCol = edgeColList[colBand];
                destRowPosList.emplace_back(lsst::geom::Point2D(endCol, endRow));
            }
            auto bottomSrcPosList = localDestToParentSrc->applyForward(destRowPosList);
            for (int colBand = 0, endBand = edgeColList.size(); colBand < endBand; ++colBand) {
                int endCol = edgeColList[colBand];
                yDeltaSrcPosList[colBand] =
                        (bottomSrcPosList[colBand] - srcPosView[endCol]) * interpInvHeight;
            }

            for (int row = prevEndRow + 1; row <= endRow; ++row) {
                typename DestImageT::x_iterator destXIter = destImage.row_begin(row);
                srcPosView[-1] += yDeltaSrcPosList[0];
                for (int colBand = 1, endBand = edgeColList.size(); colBand < endBand; ++colBand) {
                    // Next vertical interpolation band

                    int const prevEndCol = edgeColList[colBand - 1];
                    int const endCol = edgeColList[colBand];

                    // Compute xDeltaSrcPos; remember that srcPosView contains
                    // positions for this row in prevEndCol and smaller indices,
                    // and positions for the previous row for larger indices (including endCol)
                    lsst::geom::Point2D leftSrcPos = srcPosView[prevEndCol];
                    lsst::geom::Point2D rightSrcPos = srcPosView[endCol] + yDeltaSrcPosList[colBand];
                    lsst::geom::Extent2D xDeltaSrcPos = (rightSrcPos - leftSrcPos) * invWidthList[colBand];

                    for (int col = prevEndCol + 1; col <= endCol; ++col, ++destXIter) {
                        lsst::geom::Point2D leftSrcPos = srcPosView[col - 1];
                        lsst::geom::Point2D srcPos = leftSrcPos + xDeltaSrcPos;
                        double relativeArea = computeRelativeArea(srcPos, leftSrcPos, srcPosView[col]);

                        srcPosView[col] = srcPos;

                        if (warpAtOnePoint(
                                    destXIter, srcPos, relativeArea,
                                    typename image::detail::image_traits<DestImageT>::image_category())) {
                            ++numGoodPixels;
                        }
                    }  // for col
                }      // for col band
            }          // for row
        }              // while next row band

    } else {
        // No interpolation

        // prevSrcPosList = source positions from the previous row; these are used to compute pixel area;
        // to begin, compute sources positions corresponding to destination row = -1
        std::vector<lsst::geom::Point2D> destPosList;
        destPosList.reserve(1 + destWidth);
        for (int col = -1; col < destWidth; ++col) {
            destPosList.emplace_back(lsst::geom::Point2D(col, -1));
        }
        auto prevSrcPosList = localDestToParentSrc->applyForward(destPosList);

        for (int row = 0; row < destHeight; ++row) {
            destPosList.clear();
            for (int col = -1; col < destWidth; ++col) {
                destPosList.emplace_back(lsst::geom::Point2D(col, row));
            }
            auto srcPosList = localDestToParentSrc->applyForward(destPosList);

            typename DestImageT::x_iterator destXIter = destImage.row_begin(row);
            for (int col = 0; col < destWidth; ++col, ++destXIter) {
                // column index = column + 1 because the first entry in srcPosList is for column -1
                auto srcPos = srcPosList[col + 1];
                double relativeArea =
                        computeRelativeArea(srcPos, prevSrcPosList[col], prevSrcPosList[col + 1]);

                if (warpAtOnePoint(destXIter, srcPos, relativeArea,
                                   typename image::detail::image_traits<DestImageT>::image_category())) {
                    ++numGoodPixels;
                }
            }  // for col
            // move points from srcPosList to prevSrcPosList (we don't care about what ends up in srcPosList
            // because it will be reallocated anyway)
            swap(srcPosList, prevSrcPosList);
        }  // for row
    }      // if interp

    return numGoodPixels;
}

template <typename DestImageT, typename SrcImageT>
int warpCenteredImage(DestImageT &destImage, SrcImageT const &srcImage,
                      lsst::geom::LinearTransform const &linearTransform, lsst::geom::Point2D const &centerPosition,
                      WarpingControl const &control, typename DestImageT::SinglePixel padValue) {
    // force src and dest to be the same size and xy0
    if ((destImage.getWidth() != srcImage.getWidth()) || (destImage.getHeight() != srcImage.getHeight()) ||
        (destImage.getXY0() != srcImage.getXY0())) {
        std::ostringstream errStream;
        errStream << "src and dest images must have same size and xy0.";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, errStream.str());
    }

    // set the xy0 coords to 0,0 to make life easier
    SrcImageT srcImageCopy(srcImage, true);
    srcImageCopy.setXY0(0, 0);
    destImage.setXY0(0, 0);
    lsst::geom::Extent2D cLocal = lsst::geom::Extent2D(centerPosition) - lsst::geom::Extent2D(srcImage.getXY0());

    // for the affine transform, the centerPosition will not only get sheared, but also
    // moved slightly.  So we'll include a translation to move it back by an amount
    // centerPosition - translatedCenterPosition
    lsst::geom::AffineTransform affTran(linearTransform, cLocal - linearTransform(cLocal));
    std::shared_ptr<geom::TransformPoint2ToPoint2> affineTransform22 = geom::makeTransform(affTran);

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
    int n = warpImage(destImage, srcImageCopy, *affineTransform22, control, padValue);
#endif

    // fix the origin and we're done.
    destImage.setXY0(srcImage.getXY0());

    return n;
}

//
// Explicit instantiations
//
/// @cond
// may need to omit default params for EXPOSURE -- original code did that and it worked
#define EXPOSURE(PIXTYPE) image::Exposure<PIXTYPE, image::MaskPixel, image::VariancePixel>
#define MASKEDIMAGE(PIXTYPE) image::MaskedImage<PIXTYPE, image::MaskPixel, image::VariancePixel>
#define IMAGE(PIXTYPE) image::Image<PIXTYPE>
#define NL /* */

#define INSTANTIATE(DESTIMAGEPIXELT, SRCIMAGEPIXELT)                                                         \
    template int warpCenteredImage(                                                                          \
            IMAGE(DESTIMAGEPIXELT) & destImage, IMAGE(SRCIMAGEPIXELT) const &srcImage,                       \
            lsst::geom::LinearTransform const &linearTransform, lsst::geom::Point2D const &centerPosition,               \
            WarpingControl const &control, IMAGE(DESTIMAGEPIXELT)::SinglePixel padValue);                    \
    NL template int warpCenteredImage(                                                                       \
            MASKEDIMAGE(DESTIMAGEPIXELT) & destImage, MASKEDIMAGE(SRCIMAGEPIXELT) const &srcImage,           \
            lsst::geom::LinearTransform const &linearTransform, lsst::geom::Point2D const &centerPosition,               \
            WarpingControl const &control, MASKEDIMAGE(DESTIMAGEPIXELT)::SinglePixel padValue);              \
    NL template int warpImage(IMAGE(DESTIMAGEPIXELT) & destImage, IMAGE(SRCIMAGEPIXELT) const &srcImage,     \
                              geom::TransformPoint2ToPoint2 const &srcToDest, WarpingControl const &control, \
                              IMAGE(DESTIMAGEPIXELT)::SinglePixel padValue);                                 \
    NL template int warpImage(MASKEDIMAGE(DESTIMAGEPIXELT) & destImage,                                      \
                              MASKEDIMAGE(SRCIMAGEPIXELT) const &srcImage,                                   \
                              geom::TransformPoint2ToPoint2 const &srcToDest, WarpingControl const &control, \
                              MASKEDIMAGE(DESTIMAGEPIXELT)::SinglePixel padValue);                           \
    NL template int warpImage(IMAGE(DESTIMAGEPIXELT) & destImage, geom::SkyWcs const &destWcs,               \
                              IMAGE(SRCIMAGEPIXELT) const &srcImage, geom::SkyWcs const &srcWcs,             \
                              WarpingControl const &control, IMAGE(DESTIMAGEPIXELT)::SinglePixel padValue);  \
    NL template int warpImage(MASKEDIMAGE(DESTIMAGEPIXELT) & destImage, geom::SkyWcs const &destWcs,         \
                              MASKEDIMAGE(SRCIMAGEPIXELT) const &srcImage, geom::SkyWcs const &srcWcs,       \
                              WarpingControl const &control,                                                 \
                              MASKEDIMAGE(DESTIMAGEPIXELT)::SinglePixel padValue);                           \
    NL template int warpExposure(EXPOSURE(DESTIMAGEPIXELT) & destExposure,                                   \
                                 EXPOSURE(SRCIMAGEPIXELT) const &srcExposure, WarpingControl const &control, \
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
/// @endcond
}  // namespace math
}  // namespace afw
}  // namespace lsst
