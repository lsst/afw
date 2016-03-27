// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
#include <vector>

#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/geom/Point.h"

namespace lsst {
namespace afw {
namespace math {
namespace detail {

    /**
     * @brief A functor that computes one warped pixel
     */
    template<typename DestImageT, typename SrcImageT>
    class WarpAtOnePoint {
    public:
        WarpAtOnePoint(
            SrcImageT const &srcImage,
            WarpingControl const &control,
            typename DestImageT::SinglePixel padValue
        ) :
            _srcImage(srcImage),
            _kernelPtr(control.getWarpingKernel()),
            _maskKernelPtr(control.getMaskWarpingKernel()),
            _hasMaskKernel(control.getMaskWarpingKernel()),
            _kernelCtr(_kernelPtr->getCtr()),
            _maskKernelCtr(_maskKernelPtr ? _maskKernelPtr->getCtr() : lsst::afw::geom::Point2I(0, 0)),
            _growFullMask(control.getGrowFullMask()),
            _xList(_kernelPtr->getWidth()),
            _yList(_kernelPtr->getHeight()),
            _maskXList(_maskKernelPtr ? _maskKernelPtr->getWidth() : 0),
            _maskYList(_maskKernelPtr ? _maskKernelPtr->getHeight() : 0),
            _padValue(padValue),
            _srcGoodBBox(_kernelPtr->shrinkBBox(srcImage.getBBox(lsst::afw::image::LOCAL)))
        { };

        /**
         * Compute one warped pixel, Image specialization
         *
         * The Image specialization ignores the mask warping kernel, even if present
         */
        bool operator()(
            typename DestImageT::x_iterator &destXIter,
            lsst::afw::geom::Point2D const &srcPos,
            double relativeArea,
            lsst::afw::image::detail::Image_tag
        ) {
            // Compute associated source pixel index as integer and nonnegative fractional parts;
            // the latter is used to compute the remapping kernel.
            std::pair<int, double> srcIndFracX = _srcImage.positionToIndex(srcPos[0], lsst::afw::image::X);
            std::pair<int, double> srcIndFracY = _srcImage.positionToIndex(srcPos[1], lsst::afw::image::Y);
            if (srcIndFracX.second < 0) {
                ++srcIndFracX.second;
                --srcIndFracX.first;
            }
            if (srcIndFracY.second < 0) {
                ++srcIndFracY.second;
                --srcIndFracY.first;
            }
        
            if (_srcGoodBBox.contains(lsst::afw::geom::Point2I(srcIndFracX.first, srcIndFracY.first))) {
                // Offset source pixel index from kernel center to kernel corner (0, 0)
                // so we can convolveAtAPoint the pixels that overlap between source and kernel
                int srcStartX = srcIndFracX.first - _kernelCtr[0];
                int srcStartY = srcIndFracY.first - _kernelCtr[1];
        
                // Compute warped pixel
                double kSum = _setFracIndex(srcIndFracX.second, srcIndFracY.second);
        
                typename SrcImageT::const_xy_locator srcLoc = _srcImage.xy_at(srcStartX, srcStartY);
        
                *destXIter = lsst::afw::math::convolveAtAPoint<DestImageT, SrcImageT>(srcLoc, _xList, _yList);
                *destXIter *= relativeArea/kSum;
                return true;
            } else {
               // Edge pixel
                *destXIter = _padValue;
                return false;
            }
        }

        /**
         * Compute one warped pixel, MaskedImage specialization
         *
         * The MaskedImage specialization uses the mask warping kernel, if present, to compute the mask plane;
         * otherwise it uses the normal kernel to compute the mask plane.
         */
        bool operator()(
            typename DestImageT::x_iterator &destXIter,
            lsst::afw::geom::Point2D const &srcPos,
            double relativeArea,
            lsst::afw::image::detail::MaskedImage_tag
        ) {
            // Compute associated source pixel index as integer and nonnegative fractional parts;
            // the latter is used to compute the remapping kernel.
            std::pair<int, double> srcIndFracX = _srcImage.positionToIndex(srcPos[0], lsst::afw::image::X);
            std::pair<int, double> srcIndFracY = _srcImage.positionToIndex(srcPos[1], lsst::afw::image::Y);
            if (srcIndFracX.second < 0) {
                ++srcIndFracX.second;
                --srcIndFracX.first;
            }
            if (srcIndFracY.second < 0) {
                ++srcIndFracY.second;
                --srcIndFracY.first;
            }
        
            if (_srcGoodBBox.contains(lsst::afw::geom::Point2I(srcIndFracX.first, srcIndFracY.first))) {
                // Offset source pixel index from kernel center to kernel corner (0, 0)
                // so we can convolveAtAPoint the pixels that overlap between source and kernel
                int srcStartX = srcIndFracX.first - _kernelCtr[0];
                int srcStartY = srcIndFracY.first - _kernelCtr[1];
        
                // Compute warped pixel
                double kSum = _setFracIndex(srcIndFracX.second, srcIndFracY.second);
        
                typename SrcImageT::const_xy_locator srcLoc = _srcImage.xy_at(srcStartX, srcStartY);
        
                *destXIter = lsst::afw::math::convolveAtAPoint<DestImageT, SrcImageT>(srcLoc, _xList, _yList);
                *destXIter *= relativeArea/kSum;
                
                if (_hasMaskKernel) {
                    // compute mask value based on the mask kernel (replacing the value computed above)
                    int maskStartX = srcIndFracX.first - _maskKernelCtr[0];
                    int maskStartY = srcIndFracY.first - _maskKernelCtr[1];
        
                    typename SrcImageT::Mask::const_xy_locator srcMaskLoc = \
                        _srcImage.getMask()->xy_at(maskStartX, maskStartY);
            
                    typedef typename std::vector<lsst::afw::math::Kernel::Pixel>::const_iterator k_iter;
                
                    typename DestImageT::Mask::SinglePixel destMaskValue = 0;
                    for (k_iter kernelYIter = _maskYList.begin(), yEnd = _maskYList.end();
                         kernelYIter != yEnd; ++kernelYIter) {
                
                        typename DestImageT::Mask::SinglePixel destMaskValueY = 0;
                        for (k_iter kernelXIter = _maskXList.begin(), xEnd = _maskXList.end();
                             kernelXIter != xEnd; ++kernelXIter, ++srcMaskLoc.x()) {
                            typename lsst::afw::math::Kernel::Pixel const kValX = *kernelXIter;
                            if (kValX != 0) {
                                destMaskValueY |= *srcMaskLoc;
                            }
                        }
                
                        double const kValY = *kernelYIter;
                        if (kValY != 0) {
                            destMaskValue |= destMaskValueY;
                        }
                
                        srcMaskLoc += lsst::afw::image::detail::difference_type(-_maskXList.size(), 1);
                    }
        
                    destXIter.mask() = (destXIter.mask() & _growFullMask) | destMaskValue;
                }
                return true;
            } else {
               // Edge pixel
                *destXIter = _padValue;
                return false;
            }
        }
    
    private:
        /**
         * Set parameters of kernel (and mask kernel, if present) and update X and Y values
         *
         * @return sum of kernel
         */
        double _setFracIndex(double xFrac, double yFrac) {
            std::pair<double, double> srcFracInd(xFrac, yFrac);
            _kernelPtr->setKernelParameters(srcFracInd);
            double kSum = _kernelPtr->computeVectors(_xList, _yList, false);
            if (_maskKernelPtr) {
                _maskKernelPtr->setKernelParameters(srcFracInd);
                _maskKernelPtr->computeVectors(_maskXList, _maskYList, false);
            }
            return kSum;
        }

        SrcImageT _srcImage;
        PTR(lsst::afw::math::SeparableKernel) _kernelPtr;
        PTR(lsst::afw::math::SeparableKernel) _maskKernelPtr;
        bool _hasMaskKernel;
        lsst::afw::geom::Point2I _kernelCtr;
        lsst::afw::geom::Point2I _maskKernelCtr;
        lsst::afw::image::MaskPixel _growFullMask;
        std::vector<double> _xList;
        std::vector<double> _yList;
        std::vector<double> _maskXList;
        std::vector<double> _maskYList;
        typename DestImageT::SinglePixel _padValue;
        lsst::afw::geom::Box2I const _srcGoodBBox;
    };

}}}} // lsst::afw::math::detail
