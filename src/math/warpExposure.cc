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
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include "boost/shared_ptr.hpp"
#include "boost/cstdint.hpp" 
#include "boost/regex.hpp"

#include "lsst/pex/logging/Trace.h" 
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/math.h"
#include "lsst/afw/coord/Coord.h"

namespace pexExcept = lsst::pex::exceptions;
namespace pexLog = lsst::pex::logging;
namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;
namespace afwCoord = lsst::afw::coord;
namespace afwMath = lsst::afw::math;

afwMath::Kernel::Ptr afwMath::LanczosWarpingKernel::clone() const {
    return afwMath::Kernel::Ptr(new afwMath::LanczosWarpingKernel(this->getOrder()));
}

/**
* @brief get the order of the kernel
*/
int afwMath::LanczosWarpingKernel::getOrder() const {
    return this->getWidth() / 2;
}

afwMath::Kernel::Ptr afwMath::BilinearWarpingKernel::clone() const {
    return afwMath::Kernel::Ptr(new afwMath::BilinearWarpingKernel());
}

/**
* \brief Solve bilinear equation; the only permitted arguments are 0 or 1
*
* \throw lsst::pex::exceptions::InvalidParameterException if argument is not 0 or 1
*/
afwMath::Kernel::Pixel afwMath::BilinearWarpingKernel::BilinearFunction1::operator() (
    double x
) const {
#if 0 && !defined(NDEBUG)
    if (x == 0.0) {
        return 1.0 - this->_params[0];
    } else if (x == 1.0) {
        return this->_params[0];
    } else {                            // the mere presence of this check slows the call by 3 times
        std::ostringstream errStream;
        errStream << "x = " << x << "; must be 0 or 1";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, errStream.str());
    }
#else
    if (x == 0.0) {
        return 1.0 - this->_params[0];
    } else {
        return this->_params[0];
    }
#endif
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

afwMath::Kernel::Ptr afwMath::NearestWarpingKernel::clone() const {
    return afwMath::Kernel::Ptr(new afwMath::NearestWarpingKernel());
}

/**
* \brief Solve nearest neighbor equation; the only permitted arguments are 0 or 1
*
* \throw lsst::pex::exceptions::InvalidParameterException if argument is not 0 or 1
*/
afwMath::Kernel::Pixel afwMath::NearestWarpingKernel::NearestFunction1::operator() (
    double x
) const {
    if (x == 0.0) {
        return this->_params[0] < 0.5 ? 1.0 : 0.0;
    } else if (x == 1.0) {
        return this->_params[0] < 0.5 ? 0.0 : 1.0;
    } else {
        std::ostringstream errStream;
        errStream << "x = " << x << "; must be 0 or 1";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, errStream.str());
    }
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

/**
 * \brief Return a warping kernel given its name.
 *
 * Intended for use with warpImage() and warpExposure().
 *
 * Allowed names are:
 * - bilinear: return a BilinearWarpingKernel
 * - lanczos#: return a LanczosWarpingKernel of order #, e.g. lanczos4
 * - nearest: return a NearestWarpingKernel
 */
boost::shared_ptr<afwMath::SeparableKernel> afwMath::makeWarpingKernel(std::string name) {
    typedef boost::shared_ptr<afwMath::SeparableKernel> KernelPtr;
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
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
            "unknown warping kernel name: \"" + name + "\"");
    }
}

/**
 * \brief Warp (remap) one exposure to another.
 *
 * This is a convenience wrapper around warpImage(). 
 */
template<typename DestExposureT, typename SrcExposureT>
int afwMath::warpExposure(
    DestExposureT &destExposure,        ///< Remapped exposure. Wcs and xy0 are read, MaskedImage is set,
                                        ///< and Calib and Filter are copied from srcExposure.
                                        ///< All other attributes are left alone (including Detector and Psf)
    SrcExposureT const &srcExposure,    ///< Source exposure
    SeparableKernel &warpingKernel,     ///< Warping kernel; determines warping algorithm
    int const interpLength              ///< Distance over which WCS can be linearily interpolated    
    )
{
    if (!destExposure.hasWcs()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "destExposure has no Wcs");
    }
    if (!srcExposure.hasWcs()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "srcExposure has no Wcs");
    }
    typename DestExposureT::MaskedImageT mi = destExposure.getMaskedImage();
    boost::shared_ptr<afwImage::Calib> calibCopy(new afwImage::Calib(*srcExposure.getCalib()));
    destExposure.setCalib(calibCopy);
    destExposure.setFilter(srcExposure.getFilter());
    return warpImage(mi, *destExposure.getWcs(),
                     srcExposure.getMaskedImage(), *srcExposure.getWcs(), warpingKernel, interpLength);
}


/************************************************************************************************************/
namespace {


    class SrcPosFunctor {
    public:
        SrcPosFunctor() {}
        typedef boost::shared_ptr<SrcPosFunctor> Ptr;
        virtual afwGeom::Point2D operator()(int destCol, int destRow) const = 0;
    private:
    };

    class WcsSrcPosFunctor : public SrcPosFunctor {
    public:
        WcsSrcPosFunctor(
                         afwGeom::Point2D const &destXY0,    ///< xy0 of destination image
                         afwImage::Wcs const &destWcs,       ///< WCS of remapped %image
                         afwImage::Wcs const &srcWcs
                        ) :      ///< WCS of source %image
            SrcPosFunctor(),
            _destXY0(destXY0),
            _destWcs(destWcs),
            _srcWcs(srcWcs) {}
        typedef boost::shared_ptr<WcsSrcPosFunctor> Ptr;
        
        virtual afwGeom::Point2D operator()(int destCol, int destRow) const {
            double const col = afwImage::indexToPosition(destCol + _destXY0[0]);
            double const row = afwImage::indexToPosition(destRow + _destXY0[1]);
            afwGeom::Angle sky1, sky2;
            _destWcs.pixelToSky(col, row, sky1, sky2);
            return _srcWcs.skyToPixel(sky1, sky2);
        }
    private:
        afwGeom::Point2D const &_destXY0;
        afwImage::Wcs const &_destWcs;
        afwImage::Wcs const &_srcWcs;
    };

    class AffineTransformSrcPosFunctor : public SrcPosFunctor {
    public:
        // NOTE: The transform will be called to locate a *source* pixel given a *dest* pixel
        // ... so we actually want to use the *inverse* transform of the affineTransform we we're given.
        // Thus _affineTransform is initialized to affineTransform.invert()
        AffineTransformSrcPosFunctor(
                                     afwGeom::Point2D const &destXY0,    ///< xy0 of destination image 
                                     afwGeom::AffineTransform const &affineTransform
                                     ) :
            SrcPosFunctor(),
            _destXY0(destXY0),
            _affineTransform(affineTransform.invert()) {}

        virtual afwGeom::Point2D operator()(int destCol, int destRow) const {
            double const col = afwImage::indexToPosition(destCol + _destXY0[0]);
            double const row = afwImage::indexToPosition(destRow + _destXY0[1]);
            return _affineTransform(afwGeom::Point2D(col, row));
        }
    private:
        afwGeom::Point2D const &_destXY0;
        afwGeom::AffineTransform const &_affineTransform;
    };



    
    inline afwGeom::Point2D xcomputeSrcPos(
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
}

namespace {
    
    template<typename DestImageT, typename SrcImageT>
    int doWarpImage(
        DestImageT &destImage,              ///< remapped %image
        SrcImageT const &srcImage,          ///< source %image
        afwMath::SeparableKernel &warpingKernel,     ///< warping kernel; determines warping algorithm
        SrcPosFunctor const &computeSrcPos,   ///< Functor to compute source position
        int const interpLength              ///< Distance over which WCS can be linearily interpolated
            ///< 0 means no interpolation and uses an optimized branch of the code
            ///< 1 also performs no interpolation but it runs the interpolation code branch
        )
    {
    
        if (afwMath::details::isSameObject(destImage, srcImage)) {
            throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                "destImage is srcImage; cannot warp in place");
        }
        int numGoodPixels = 0;
    
        typedef afwImage::Image<afwMath::Kernel::Pixel> KernelImageT;
        
        // Compute borders; use to prevent applying kernel outside of srcImage
        int const kernelWidth = warpingKernel.getWidth();
        int const kernelHeight = warpingKernel.getHeight();
        int const kernelCtrX = warpingKernel.getCtrX();
        int const kernelCtrY = warpingKernel.getCtrY();
    
        // Get the source MaskedImage and a pixel accessor to it.
        int const srcWidth = srcImage.getWidth();
        int const srcHeight = srcImage.getHeight();
        pexLog::TTrace<3>("lsst.afw.math.warp", "source image width=%d; height=%d", srcWidth, srcHeight);
    
        int const destWidth = destImage.getWidth();
        int const destHeight = destImage.getHeight();
        
        pexLog::TTrace<3>("lsst.afw.math.warp", "remap image width=%d; height=%d", destWidth, destHeight);
    
        typename DestImageT::SinglePixel const edgePixel = afwMath::edgePixel<DestImageT>(
            typename afwImage::detail::image_traits<DestImageT>::image_category()
        );
        
        std::vector<double> kernelXList(kernelWidth);
        std::vector<double> kernelYList(kernelHeight);
        
        afwGeom::Box2I srcGoodBBox = warpingKernel.shrinkBBox(srcImage.getBBox(afwImage::LOCAL));
    
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
                //afwGeom::Point2D rightSrcPos = computeSrcPos(endCol, -1, destXY0, destWcs, srcWcs);
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
                    //afwGeom::Point2D bottomSrcPos = computeSrcPos(endCol, endRow, destXY0, destWcs, srcWcs);
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
            
                            // Compute associated source pixel index as integer and nonnegative fractional parts;
                            // the latter is used to compute the remapping kernel.
                            std::pair<int, double> srcIndFracX = srcImage.positionToIndex(srcPos[0], afwImage::X);
                            std::pair<int, double> srcIndFracY = srcImage.positionToIndex(srcPos[1], afwImage::Y);
                            if (srcIndFracX.second < 0) {
                                ++srcIndFracX.second;
                                --srcIndFracX.first;
                            }
                            if (srcIndFracY.second < 0) {
                                ++srcIndFracY.second;
                                --srcIndFracY.first;
                            }
                            
                            if (srcGoodBBox.contains(afwGeom::Point2I(srcIndFracX.first, srcIndFracY.first))) {
                                 ++numGoodPixels;
            
                                // Offset source pixel index from kernel center to kernel corner (0, 0)
                                // so we can convolveAtAPoint the pixels that overlap between source and kernel
                                srcIndFracX.first -= kernelCtrX;
                                srcIndFracY.first -= kernelCtrY;
                                    
                                // Compute warped pixel
                                std::pair<double, double> srcFracInd(srcIndFracX.second, srcIndFracY.second);
                                warpingKernel.setKernelParameters(srcFracInd);
                                double kSum = warpingKernel.computeVectors(kernelXList, kernelYList, false);
                
                                typename SrcImageT::const_xy_locator srcLoc =
                                    srcImage.xy_at(srcIndFracX.first, srcIndFracY.first);
                                
                                *destXIter = afwMath::convolveAtAPoint<DestImageT,SrcImageT>(
                                    srcLoc, kernelXList, kernelYList);
                                *destXIter *= relativeArea/kSum;
                            } else {
                               // Edge pixel pixel
                                *destXIter = edgePixel;
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
                //srcPosView[col] = computeSrcPos(col, -1, destXY0, destWcs, srcWcs);
                srcPosView[col] = computeSrcPos(col, -1);
            }
            
            for (int row = 0; row < destHeight; ++row) {
                typename DestImageT::x_iterator destXIter = destImage.row_begin(row);
                
                //srcPosView[-1] = computeSrcPos(-1, row, destXY0, destWcs, srcWcs);
                srcPosView[-1] = computeSrcPos(-1, row);
                
                for (int col = 0; col < destWidth; ++col, ++destXIter) {
                    //afwGeom::Point2D srcPos = computeSrcPos(col, row, destXY0, destWcs, srcWcs);
                    afwGeom::Point2D srcPos = computeSrcPos(col, row);
                    double relativeArea = computeRelativeArea(srcPos, srcPosView[col-1], srcPosView[col]);
                    srcPosView[col] = srcPos;
    
                    // Compute associated source pixel index as integer and nonnegative fractional parts;
                    // the latter is used to compute the remapping kernel.
                    std::pair<int, double> srcIndFracX = srcImage.positionToIndex(srcPos[0], afwImage::X);
                    std::pair<int, double> srcIndFracY = srcImage.positionToIndex(srcPos[1], afwImage::Y);
                    if (srcIndFracX.second < 0) {
                        ++srcIndFracX.second;
                        --srcIndFracX.first;
                    }
                    if (srcIndFracY.second < 0) {
                        ++srcIndFracY.second;
                        --srcIndFracY.first;
                    }
                    
                    if (srcGoodBBox.contains(afwGeom::Point2I(srcIndFracX.first, srcIndFracY.first))) {
                         ++numGoodPixels;
    
                        // Offset source pixel index from kernel center to kernel corner (0, 0)
                        // so we can convolveAtAPoint the pixels that overlap between source and kernel
                        srcIndFracX.first -= kernelCtrX;
                        srcIndFracY.first -= kernelCtrY;
                            
                        // Compute warped pixel
                        std::pair<double, double> srcFracInd(srcIndFracX.second, srcIndFracY.second);
                        warpingKernel.setKernelParameters(srcFracInd);
                        double kSum = warpingKernel.computeVectors(kernelXList, kernelYList, false);
        
                        typename SrcImageT::const_xy_locator srcLoc =
                            srcImage.xy_at(srcIndFracX.first, srcIndFracY.first);
                        
                        *destXIter = afwMath::convolveAtAPoint<DestImageT,SrcImageT>(
                            srcLoc, kernelXList, kernelYList);
                        *destXIter *= relativeArea/kSum;
                    } else {
                       // Edge pixel pixel
                        *destXIter = edgePixel;
                    }
                }   // for col
            }   // for row
        } // if interp
    
        return numGoodPixels;
    }
} // namespace

/**
 * \brief Remap an Image or MaskedImage to a new Wcs. See also convenience function
 * warpExposure() to warp an Exposure.
 *
 * Edge pixels of destImage are set to the value returned by edgePixel().
 * These are pixels whose data comes from too near the edge of srcImage, or misses srcImage entirely.
 *
 * \return the number of valid pixels in destImage (those that are not edge pixels).
 *
 * \b Warping \b Kernels:
 *
 * This function requires a warping kernel to perform the interpolation.
 * Available warping kernels include:
 * - BilinearWarpingKernel
 * - LanczosWarpingKernel
 * - NearestWarpingKernel (nearest neighbor)
 *
 * makeWarpingKernel() is a handy factory function for constructing a warping kernel given its name.
 *
 * A warping kernel is a subclass of SeparableKernel with the following properties:
 * - It has two parameters: fractional x and fractional row position on the source %image.
 *   The fractional position for each axis is in the range [0, 1):
 *   - 0 if the position on the source along that axis is on the center of the pixel.
 *   - 0.999... if the position on the source along that axis is almost on the center of the next pixel.
 * - It almost always has even width and height (which is unusual for a kernel) and a center index of
 *   (width/2, /height/2). This is because the kernel is used to map source positions that range from
 *   centered on on pixel (width/2, height/2) to nearly centered on pixel (width/2 + 1, height/2 + 1).
 *
 * \b Algorithm Without Interpolation:
 *
 * For each integer pixel position in the remapped Exposure:
 * - The associated pixel position on srcImage is determined using the destination and source WCS.
 * - The warping kernel's parameters are set based on the fractional part of the pixel position on srcImage
 * - The warping kernel is applied to srcImage at the integer portion of the pixel position
 *   to compute the remapped pixel value
 * - A flux-conservation factor is determined from the source and destination WCS
 *   and is applied to the remapped pixel
 *
 * The scaling of intensity for relative area of source and destination uses two minor approximations:
 * - The area of the sky marked out by a pixel on the destination %image
 *   corresponds to a parallellogram on the source %image.
 * - The area varies slowly enough across the %image that we can get away with computing
 *   the source area shifted by half a pixel up and to the left of the true area.
 *
 * \b Algorithm With Interpolation:
 *
 * Interpolation simply reduces the number of times WCS is used to map between destination and source
 * pixel position. This computation is only made at a grid of points on the destination image,
 * separated by interpLen pixels along rows and columns. All other source pixel positions are determined
 * by linear interpolation between those grid points. Everything else remains the same.
 *
 * \throw lsst::pex::exceptions::InvalidParameterException if destImage is srcImage
 *
 * \todo Should support an additional color-based position correction in the remapping (differential chromatic
 *   refraction). This can be done either object-by-object or pixel-by-pixel.
 *
 * \todo Need to deal with oversampling and/or weight maps. If done we can use faster kernels than sinc.
 */
template<typename DestImageT, typename SrcImageT>
int afwMath::warpImage(
    DestImageT &destImage,              ///< remapped %image
    afwImage::Wcs const &destWcs,       ///< WCS of remapped %image
    SrcImageT const &srcImage,          ///< source %image
    afwImage::Wcs const &srcWcs,        ///< WCS of source %image
    SeparableKernel &warpingKernel,     ///< warping kernel; determines warping algorithm
    int const interpLength              ///< Distance over which WCS can be linearily interpolated
        ///< 0 means no interpolation and uses an optimized branch of the code
        ///< 1 also performs no interpolation but it runs the interpolation code branch
    )
{
    afwGeom::Point2D const destXY0(destImage.getXY0());
    WcsSrcPosFunctor const computeSrcPos(destXY0, destWcs, srcWcs);
    return doWarpImage(destImage, srcImage, warpingKernel, computeSrcPos, interpLength);
}


template<typename DestImageT, typename SrcImageT>
int afwMath::warpImage(
    DestImageT &destImage,                      ///< remapped %image
    SrcImageT const &srcImage,                  ///< source %image
    SeparableKernel &warpingKernel,             ///< warping kernel; determines warping algorithm
    afwGeom::AffineTransform const &affineTransform, ///< affine transformation to apply
    int const interpLength                      ///< Distance over which WCS can be linearily interpolated
        ///< 0 means no interpolation and uses an optimized branch of the code
        ///< 1 also performs no interpolation but it runs the interpolation code branch
                      )
{
    afwGeom::Point2D const destXY0(destImage.getXY0());
    AffineTransformSrcPosFunctor const computeSrcPos(destXY0, affineTransform);
    return doWarpImage(destImage, srcImage, warpingKernel, computeSrcPos, interpLength);
}


template<typename DestImageT, typename SrcImageT>
int afwMath::warpCenteredImage(
    DestImageT &destImage,                      ///< remapped %image
    SrcImageT const &srcImage,                  ///< source %image
    SeparableKernel &warpingKernel,             ///< warping kernel; determines warping algorithm
    afwGeom::LinearTransform const &linearTransform, ///< linear transformation to apply
    afwGeom::Point2D const &centerPixel         ///< pixel corresponding to location of linearTransform
                      )
{

    // force src and dest to be the same size and xy0
    if (
        (destImage.getWidth() != srcImage.getWidth()) ||
        (destImage.getHeight() != srcImage.getHeight()) ||
        (destImage.getXY0() != srcImage.getXY0())
       ) {
        std::ostringstream errStream;
        errStream << "src and dest images must have same size and xy0.";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, errStream.str());
    }
    
    // set the xy0 coords to 0,0 to make life easier
    SrcImageT srcImageCopy(srcImage, true);
    srcImageCopy.setXY0(0, 0);
    destImage.setXY0(0, 0);
    afwGeom::Extent2D cLocal = afwGeom::Extent2D(centerPixel) - afwGeom::Extent2D(srcImage.getXY0());

    // for the affine transform, the centerPixel will not only get sheared, but also
    // moved slightly.  So we'll include a translation to move it back by an amount
    // centerPixel - translatedCenterPixel
    afwGeom::AffineTransform affTran(linearTransform, cLocal - linearTransform(cLocal));
    
    // now warp
    int n = warpImage(destImage, srcImageCopy, warpingKernel, affTran);

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
        SeparableKernel &warpingKernel,                                 \
        afwGeom::LinearTransform const &linearTransform,                \
        afwGeom::Point2D const &centerPixel); NL \
    template int afwMath::warpCenteredImage(                                    \
        MASKEDIMAGE(DESTIMAGEPIXELT) &destImage, \
        MASKEDIMAGE(SRCIMAGEPIXELT) const &srcImage, \
        SeparableKernel &warpingKernel,                                 \
        afwGeom::LinearTransform const &linearTransform,                \
        afwGeom::Point2D const &centerPixel); NL \
    template int afwMath::warpImage( \
        IMAGE(DESTIMAGEPIXELT) &destImage, \
        IMAGE(SRCIMAGEPIXELT) const &srcImage, \
        SeparableKernel &warpingKernel,                                 \
        afwGeom::AffineTransform const &affineTransform,  int const interpLength); NL \
    template int afwMath::warpImage(                                    \
        MASKEDIMAGE(DESTIMAGEPIXELT) &destImage, \
        MASKEDIMAGE(SRCIMAGEPIXELT) const &srcImage, \
        SeparableKernel &warpingKernel,                                 \
        afwGeom::AffineTransform const &affineTransform,  int const interpLength); NL \
    template int afwMath::warpImage(                                    \
        IMAGE(DESTIMAGEPIXELT) &destImage,  \
        afwImage::Wcs const &destWcs,          \
        IMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwImage::Wcs const &srcWcs, \
        SeparableKernel &warpingKernel, int const interpLength); NL    \
    template int afwMath::warpImage( \
        MASKEDIMAGE(DESTIMAGEPIXELT) &destImage, \
        afwImage::Wcs const &destWcs, \
        MASKEDIMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwImage::Wcs const &srcWcs, \
        SeparableKernel &warpingKernel, int const interpLength); NL    \
    template int afwMath::warpExposure(                                \
        EXPOSURE(DESTIMAGEPIXELT) &destExposure, \
        EXPOSURE(SRCIMAGEPIXELT) const &srcExposure, \
        SeparableKernel &warpingKernel, int const interpLength);




INSTANTIATE(double, double)
INSTANTIATE(double, float)
INSTANTIATE(double, int)
INSTANTIATE(double, boost::uint16_t)
INSTANTIATE(float, float)
INSTANTIATE(float, int)
INSTANTIATE(float, boost::uint16_t)
INSTANTIATE(int, int)
INSTANTIATE(boost::uint16_t, boost::uint16_t)
/// \endcond
