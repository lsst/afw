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
#include <vector>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/geom/Point.h"

namespace lsst {
namespace afw {
namespace math {
namespace detail {

    /**
     * @brief A class to manage a warping kernel and optional mask warping kernel
     *
     * Designed to support computeOneWarpedPixel
     */
    class WarpingKernelInfo {
    public:
        WarpingKernelInfo(
            lsst::afw::math::SeparableKernel::Ptr kernelPtr,    ///< warping kernel (required)
            lsst::afw::math::SeparableKernel::Ptr maskKernelPtr ///< mask warping kernel (optional)
        ) :
            _kernelPtr(kernelPtr),
            _maskKernelPtr(maskKernelPtr),
            _xList(kernelPtr->getWidth()),
            _yList(kernelPtr->getHeight()),
            _maskXList(maskKernelPtr ? maskKernelPtr->getWidth() : 0),
            _maskYList(maskKernelPtr ? maskKernelPtr->getHeight() : 0)
        {
            // test that border of mask kernel <= border of kernel
            if (maskKernelPtr) {
                // compute bounding boxes with 0,0 at kernel center
                // and make sure kernel bbox includes mask kernel bbox
                lsst::afw::geom::Box2I kernelBBox = lsst::afw::geom::Box2I(
                    lsst::afw::geom::Point2I(0, 0) - lsst::afw::geom::Extent2I(kernelPtr->getCtr()),
                    kernelPtr->getDimensions()
                );
                lsst::afw::geom::Box2I maskKernelBBox = lsst::afw::geom::Box2I(
                    lsst::afw::geom::Point2I(0, 0) - lsst::afw::geom::Extent2I(maskKernelPtr->getCtr()),
                    maskKernelPtr->getDimensions()
                );
                if (!kernelBBox.contains(maskKernelBBox)) {
                    throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                        "mask kernel extends beyond kernel on at least one side of center");
                }
            }
        };
        
        /**
         * Set parameters of kernel (and mask kernel, if present) and update X and Y values
         *
         * @return sum of kernel
         */
        double setFracIndex(double xFrac, double yFrac) {
            std::pair<double, double> srcFracInd(xFrac, yFrac);
            _kernelPtr->setKernelParameters(srcFracInd);
            double kSum = _kernelPtr->computeVectors(_xList, _yList, false);
            if (_maskKernelPtr) {
                _maskKernelPtr->setKernelParameters(srcFracInd);
                _maskKernelPtr->computeVectors(_maskXList, _maskYList, false);
            }
            return kSum;
        }
        
        /**
         * @brief Get center index of kernel
         */
        lsst::afw::geom::Point2I getKernelCtr() const {
            return _kernelPtr->getCtr();
        }

        /**
         * @brief Get center index of mask kernel
         *
         * @throw pex_exception NotFoundException if no mask kernel
         */
        lsst::afw::geom::Point2I getMaskKernelCtr() const {
            if (not _maskKernelPtr) {
                throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException, "No mask kernel");
            }
            return _kernelPtr->getCtr();
        }
        
        /**
         * @brief Is there is a mask kernel?
         */
        bool hasMaskKernel() const { return bool(_maskKernelPtr); }

        /**
         * @brief get kernel X values
         */        
        std::vector<double> const & getXList() const { return _xList; }

        /**
         * @brief get kernel Y values
         */        
        std::vector<double> const & getYList() const { return _xList; }

        /**
         * @brief get mask kernel X values (an empty list if no mask kernel)
         */        
        std::vector<double> const & getMaskXList() const { return _maskXList; }

        /**
         * @brief get mask kernel Y values (an empty list if no mask kernel)
         */        
        std::vector<double> const & getMaskYList() const { return _maskYList; }
    
    private:
        lsst::afw::math::SeparableKernel::Ptr _kernelPtr;
        lsst::afw::math::SeparableKernel::Ptr _maskKernelPtr;
        std::vector<double> _xList;
        std::vector<double> _yList;
        std::vector<double> _maskXList;
        std::vector<double> _maskYList;
    };
    
    
    /**
     * @brief Compute one warped pixel, Image version
     *
     * This is the Image version; it ignores the mask kernel.
     */
    template<typename DestImageT, typename SrcImageT>
    bool computeOneWarpedPixel(
        typename DestImageT::x_iterator &destXIter, ///< output pixel as an x iterator
        WarpingKernelInfo &kernelInfo,              ///< information about the warping kernel
        SrcImageT const &srcImage,                  ///< source image
        lsst::afw::geom::Box2I const &srcGoodBBox,  ///< good region of source image
        lsst::afw::geom::Point2D const &srcPos,     ///< pixel position on source image at which to warp
        double relativeArea,    ///< output/input area a pixel covers on the sky
        typename DestImageT::SinglePixel const &padValue,
            ///< result if warped pixel is undefined (off the edge edge)
        lsst::afw::image::detail::Image_tag
            ///< lsst::afw::image::detail::image_traits<ImageT>::image_category()
    );
    
    /**
     * @brief Compute one warped pixel, MaskedImage version
     *
     * This is the MaskedImage version; it uses the mask kernel, if present, to compute the mask pixel.
     */
    template<typename DestImageT, typename SrcImageT>
    bool computeOneWarpedPixel(
        typename DestImageT::x_iterator &destXIter, ///< output pixel as an x iterator
        WarpingKernelInfo &kernelInfo,              ///< information about the warping kernel
        SrcImageT const &srcImage,                  ///< source image
        lsst::afw::geom::Box2I const &srcGoodBBox,  ///< good region of source image
        lsst::afw::geom::Point2D const &srcPos,     ///< pixel position on source image at which to warp
        double relativeArea,    ///< output/input area a pixel covers on the sky
        typename DestImageT::SinglePixel const &padValue,
            ///< result if warped pixel is undefined (off the edge edge)
        lsst::afw::image::detail::MaskedImage_tag
            ///< lsst::afw::image::detail::image_traits<MaskedImageT>::image_category()
    );

}}}} // lsst::afw::math::detail
