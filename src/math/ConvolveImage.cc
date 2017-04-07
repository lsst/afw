// -*- LSST-C++ -*-

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
 * Definition of functions declared in ConvolveImage.h
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <sstream>
#include <limits>
#include <vector>
#include <string>

#include "lsst/pex/exceptions.h"
#include "lsst/log/Log.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/detail/Convolve.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;
namespace mathDetail = lsst::afw::math::detail;

namespace {

    /**
    * Set the edge pixels of a convolved Image based on size of the convolution kernel used
    *
    * Separate specializations for Image and MaskedImage are required to set the EDGE bit of the Mask plane
    * (if there is one) when doCopyEdge is true.
    *
    * @param[out] outImage %image whose edge pixels are to be set
    * @param[in] kernel convolution kernel; kernel size is used to determine the edge
    * @param[in] inImage %image whose edge pixels are to be copied; ignored if doCopyEdge is false
    * @param[in] doCopyEdge if false (default), set edge pixels to the standard edge pixel; if true,
    *                   copy edge pixels from input and set EDGE bit of mask
    */
    template <typename OutImageT, typename InImageT>
    inline void setEdgePixels(
            OutImageT& outImage,
            afwMath::Kernel const &kernel,
            InImageT const &inImage,
            bool doCopyEdge,
            lsst::afw::image::detail::Image_tag)
    {
        const unsigned int imWidth = outImage.getWidth();
        const unsigned int imHeight = outImage.getHeight();
        const unsigned int kWidth = kernel.getWidth();
        const unsigned int kHeight = kernel.getHeight();
        const unsigned int kCtrX = kernel.getCtrX();
        const unsigned int kCtrY = kernel.getCtrY();

        const typename OutImageT::SinglePixel edgePixel = afwMath::edgePixel<OutImageT>(
            typename lsst::afw::image::detail::image_traits<OutImageT>::image_category()
        );
        std::vector<afwGeom::Box2I> bboxList;

        // create a list of bounding boxes describing edge regions, in this order:
        // bottom edge, top edge (both edge to edge),
        // left edge, right edge (both omitting pixels already in the bottom and top edge regions)
        int const numHeight = kHeight - (1 + kCtrY);
        int const numWidth = kWidth - (1 + kCtrX);
        bboxList.push_back(
            afwGeom::Box2I(afwGeom::Point2I(0, 0), afwGeom::Extent2I(imWidth, kCtrY))
        );
        bboxList.push_back(
            afwGeom::Box2I(afwGeom::Point2I(0, imHeight - numHeight), afwGeom::Extent2I(imWidth, numHeight))
        );
        bboxList.push_back(
            afwGeom::Box2I(afwGeom::Point2I(0, kCtrY), afwGeom::Extent2I(kCtrX, imHeight + 1 - kHeight))
        );
        bboxList.push_back(
            afwGeom::Box2I(afwGeom::Point2I(imWidth - numWidth, kCtrY), afwGeom::Extent2I(numWidth, imHeight + 1 - kHeight))
        );

        for (std::vector<afwGeom::Box2I>::const_iterator bboxIter = bboxList.begin();
            bboxIter != bboxList.end(); ++bboxIter
        ) {
            OutImageT outView(outImage, *bboxIter, afwImage::LOCAL);
            if (doCopyEdge) {
                // note: set only works with data of the same type
                // so convert the input image to output format
                outView.assign(OutImageT(InImageT(inImage, *bboxIter, afwImage::LOCAL), true));
            } else {
                outView = edgePixel;
            }
        }
    }

    /**
    * Set the edge pixels of a convolved MaskedImage based on size of the convolution kernel used
    *
    * Separate specializations for Image and MaskedImage are required to set the EDGE bit of the Mask plane
    * (if there is one) when doCopyEdge is true.
    *
    * @param[out] outImage %image whose edge pixels are to be set
    * @param[in] kernel convolution kernel; kernel size is used to determine the edge
    * @param[in] inImage  %image whose edge pixels are to be copied; ignored if doCopyEdge false
    * @param[in] doCopyEdge if false (default), set edge pixels to the standard edge pixel; if true, copy \
    *                       edge pixels from input and set EDGE bit of mask
    */
    template <typename OutImageT, typename InImageT>
    inline void setEdgePixels(
            OutImageT& outImage,
            afwMath::Kernel const &kernel,
            InImageT const &inImage,
            bool doCopyEdge,
            lsst::afw::image::detail::MaskedImage_tag)
    {
        const unsigned int imWidth = outImage.getWidth();
        const unsigned int imHeight = outImage.getHeight();
        const unsigned int kWidth = kernel.getWidth();
        const unsigned int kHeight = kernel.getHeight();
        const unsigned int kCtrX = kernel.getCtrX();
        const unsigned int kCtrY = kernel.getCtrY();

        const typename OutImageT::SinglePixel edgePixel = afwMath::edgePixel<OutImageT>(
            typename lsst::afw::image::detail::image_traits<OutImageT>::image_category()
        );
        std::vector<afwGeom::Box2I> bboxList;

        // create a list of bounding boxes describing edge regions, in this order:
        // bottom edge, top edge (both edge to edge),
        // left edge, right edge (both omitting pixels already in the bottom and top edge regions)
        int const numHeight = kHeight - (1 + kCtrY);
        int const numWidth = kWidth - (1 + kCtrX);
        bboxList.push_back(
            afwGeom::Box2I(
                afwGeom::Point2I(0, 0),
                afwGeom::Extent2I(imWidth, kCtrY)
            )
        );
        bboxList.push_back(
            afwGeom::Box2I(
                afwGeom::Point2I(0, imHeight - numHeight),
                afwGeom::Extent2I(imWidth, numHeight)
            )
        );
        bboxList.push_back(
            afwGeom::Box2I(
                afwGeom::Point2I(0, kCtrY),
                afwGeom::Extent2I(kCtrX, imHeight + 1 - kHeight)
            )
        );
        bboxList.push_back(
            afwGeom::Box2I(
                afwGeom::Point2I(imWidth - numWidth, kCtrY),
                afwGeom::Extent2I(numWidth, imHeight + 1 - kHeight)
            )
        );

        afwImage::MaskPixel const edgeMask = afwImage::Mask<afwImage::MaskPixel>::getPlaneBitMask("EDGE");
        for (std::vector<afwGeom::Box2I>::const_iterator bboxIter = bboxList.begin();
            bboxIter != bboxList.end(); ++bboxIter) {
            OutImageT outView(outImage, *bboxIter, afwImage::LOCAL);
            if (doCopyEdge) {
                // note: set only works with data of the same type
                // so convert the input image to output format
                outView.assign(OutImageT(InImageT(inImage, *bboxIter, afwImage::LOCAL), true));
                *(outView.getMask()) |= edgeMask;
            } else {
                outView = edgePixel;
            }
        }
    }

}   // anonymous namespace

template <typename OutImageT, typename InImageT>
void afwMath::scaledPlus(
        OutImageT &outImage,
        double c1,
        InImageT const &inImage1,
        double c2,
        InImageT const &inImage2)
{
    if (outImage.getDimensions() != inImage1.getDimensions()) {
        std::ostringstream os;
        os << "outImage dimensions = ( " << outImage.getWidth() << ", " << outImage.getHeight()
            << ") != (" << inImage1.getWidth() << ", " << inImage1.getHeight()
            << ") = inImage1 dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    } else if (inImage1.getDimensions() != inImage2.getDimensions()) {
        std::ostringstream os;
        os << "inImage1 dimensions = ( " << inImage1.getWidth() << ", " << inImage1.getHeight()
            << ") != (" << inImage2.getWidth() << ", " << inImage2.getHeight()
            << ") = inImage2 dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }

    typedef typename InImageT::const_x_iterator InConstXIter;
    typedef typename OutImageT::x_iterator OutXIter;
    for (int y = 0; y != inImage1.getHeight(); ++y) {
        InConstXIter const end1 = inImage1.row_end(y);
        InConstXIter inIter1 = inImage1.row_begin(y);
        InConstXIter inIter2 = inImage2.row_begin(y);
        OutXIter outIter = outImage.row_begin(y);
        for (; inIter1 != end1; ++inIter1, ++inIter2, ++outIter) {
            *outIter = (*inIter1 * c1) + (*inIter2 * c2);
        }
    }
}

template <typename OutImageT, typename InImageT, typename KernelT>
void afwMath::convolve(
        OutImageT& convolvedImage,
        InImageT const& inImage,
        KernelT const& kernel,
        ConvolutionControl const& convolutionControl)
{
    mathDetail::basicConvolve(convolvedImage, inImage, kernel, convolutionControl);
    setEdgePixels(convolvedImage, kernel, inImage, convolutionControl.getDoCopyEdge(),
        typename lsst::afw::image::detail::image_traits<OutImageT>::image_category()
    );
    convolvedImage.setXY0(inImage.getXY0());
}

template <typename OutImageT, typename InImageT, typename KernelT>
void afwMath::convolve(
        OutImageT& convolvedImage,
        InImageT const& inImage,
        KernelT const& kernel,
        bool doNormalize,
        bool doCopyEdge)
{
    ConvolutionControl convolutionControl;
    convolutionControl.setDoNormalize(doNormalize);
    convolutionControl.setDoCopyEdge(doCopyEdge);
    afwMath::convolve(convolvedImage, inImage, kernel, convolutionControl);
}




/// @cond
/*
 * Explicit instantiation of all convolve functions.
 *
 * This code needs to be compiled with full optimization, and there's no reason why
 * it should be instantiated in the swig wrappers.
 */
#define IMAGE(PIXTYPE) afwImage::Image<PIXTYPE>
#define MASKEDIMAGE(PIXTYPE) afwImage::MaskedImage<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
//
// Next a macro to generate needed instantiations for IMGMACRO (either IMAGE or MASKEDIMAGE)
// and the specified pixel types
//
/* NL's a newline for debugging -- don't define it and say
 g++ -C -E -I$(eups list -s -d boost)/include Convolve.cc | perl -pe 's| *NL *|\n|g'
*/
#define NL /* */
//
// Instantiate one kernel-specific specializations of convolution functions for Image or MaskedImage
// IMGMACRO = IMAGE or MASKEDIMAGE
// KERNELTYPE = a kernel class
//
#define INSTANTIATE_IM_OR_MI_KERNEL(IMGMACRO, OUTPIXTYPE, INPIXTYPE, KERNELTYPE) \
    template void afwMath::convolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, KERNELTYPE const&, bool, bool); NL \
    template void afwMath::convolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, KERNELTYPE const&, ConvolutionControl const&); NL
//
// Instantiate Image or MaskedImage versions of all functions defined in this file.
// Call INSTANTIATE_IM_OR_MI_KERNEL once for each kernel class.
// IMGMACRO = IMAGE or MASKEDIMAGE
//
#define INSTANTIATE_IM_OR_MI(IMGMACRO, OUTPIXTYPE, INPIXTYPE) \
    template void afwMath::scaledPlus( \
        IMGMACRO(OUTPIXTYPE)&, double, IMGMACRO(INPIXTYPE) const&, double, IMGMACRO(INPIXTYPE) const&); NL \
    INSTANTIATE_IM_OR_MI_KERNEL(IMGMACRO, OUTPIXTYPE, INPIXTYPE, afwMath::AnalyticKernel) \
    INSTANTIATE_IM_OR_MI_KERNEL(IMGMACRO, OUTPIXTYPE, INPIXTYPE, afwMath::DeltaFunctionKernel) \
    INSTANTIATE_IM_OR_MI_KERNEL(IMGMACRO, OUTPIXTYPE, INPIXTYPE, afwMath::FixedKernel) \
    INSTANTIATE_IM_OR_MI_KERNEL(IMGMACRO, OUTPIXTYPE, INPIXTYPE, afwMath::LinearCombinationKernel) \
    INSTANTIATE_IM_OR_MI_KERNEL(IMGMACRO, OUTPIXTYPE, INPIXTYPE, afwMath::SeparableKernel) \
    INSTANTIATE_IM_OR_MI_KERNEL(IMGMACRO, OUTPIXTYPE, INPIXTYPE, afwMath::Kernel) \
//
// Instantiate all functions defined in this file for one specific output and input pixel type
//
#define INSTANTIATE(OUTPIXTYPE, INPIXTYPE) \
    INSTANTIATE_IM_OR_MI(IMAGE,       OUTPIXTYPE, INPIXTYPE) \
    INSTANTIATE_IM_OR_MI(MASKEDIMAGE, OUTPIXTYPE, INPIXTYPE)
//
// Instantiate all functions defined in this file
//
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
