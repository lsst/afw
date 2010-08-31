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
 
/**
 * @file
 *
 * @brief Definition of basicConvolve and convolveWithBruteForce functions declared in detail/ConvolveImage.h
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include "boost/cstdint.hpp" 

#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image.h"
#include "lsst/afw/math.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/geom/deprecated.h"
#include "lsst/afw/math/detail/Convolve.h"

namespace pexExcept = lsst::pex::exceptions;
namespace pexLog = lsst::pex::logging;
namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;
namespace mathDetail = lsst::afw::math::detail;

namespace {

    /*
     * @brief Compute the dot product of a kernel row or column and the overlapping portion of an %image
     *
     * @return computed dot product
     *
     * The pixel computed belongs at position imageIter + kernel center.
     *
     * @todo get rid of KernelPixelT parameter if possible by not computing local variable kVal,
     * or by using iterator traits:
     *     typedef typename std::iterator_traits<KernelIterT>::value_type KernelPixel;
     * Unfortunately, in either case compilation fails with this sort of message:
\verbatim
include/lsst/afw/image/Pixel.h: In instantiation of ‘lsst::afw::image::pixel::exprTraits<boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > > >’:
include/lsst/afw/image/Pixel.h:385:   instantiated from ‘lsst::afw::image::pixel::BinaryExpr<lsst::afw::image::pixel::Pixel<int, short unsigned int, float>, boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > >, std::multiplies<int>, lsst::afw::image::pixel::bitwise_or<short unsigned int>, lsst::afw::image::pixel::variance_multiplies<float> >’
src/math/ConvolveImage.cc:59:   instantiated from ‘OutPixelT<unnamed>::kernelDotProduct(ImageIterT, KernelIterT, int) [with OutPixelT = lsst::afw::image::pixel::SinglePixel<int, short unsigned int, float>, ImageIterT = lsst::afw::image::MaskedImage<int, short unsigned int, float>::const_MaskedImageIterator<boost::gil::gray32s_pixel_t*, boost::gil::gray16_pixel_t*, boost::gil::gray32f_noscale_pixel_t*>, KernelIterT = const boost::gil::gray64f_noscalec_pixel_t*]’
src/math/ConvolveImage.cc:265:   instantiated from ‘void lsst::afw::math::basicConvolve(OutImageT&, const InImageT&, const lsst::afw::math::Kernel&, bool) [with OutImageT = lsst::afw::image::MaskedImage<int, short unsigned int, float>, InImageT = lsst::afw::image::MaskedImage<int, short unsigned int, float>]’
src/math/ConvolveImage.cc:451:   instantiated from ‘void lsst::afw::math::convolve(OutImageT&, const InImageT&, const KernelT&, bool, int) [with OutImageT = lsst::afw::image::MaskedImage<int, short unsigned int, float>, InImageT = lsst::afw::image::MaskedImage<int, short unsigned int, float>, KernelT = lsst::afw::math::AnalyticKernel]’
src/math/ConvolveImage.cc:587:   instantiated from here
include/lsst/afw/image/Pixel.h:210: error: no type named ‘ImagePixelT’ in ‘struct boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > >’
include/lsst/afw/image/Pixel.h:211: error: no type named ‘MaskPixelT’ in ‘struct boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > >’
include/lsst/afw/image/Pixel.h:212: error: no type named ‘VariancePixelT’ in ‘struct boost::gil::pixel<double, boost::gil::layout<boost::mpl::vector1<boost::gil::gray_color_t>, boost::mpl::range_c<int, 0, 1> > >’
\endverbatim
     */
    template <typename OutPixelT, typename ImageIterT, typename KernelIterT, typename KernelPixelT>
    inline OutPixelT kernelDotProduct(
            ImageIterT imageIter,       ///< start of input %image that overlaps kernel vector
            KernelIterT kernelIter,     ///< start of kernel vector
            int kWidth)                 ///< width of kernel
    {
        OutPixelT outPixel(0);
        for (int x = 0; x < kWidth; ++x, ++imageIter, ++kernelIter) {
            KernelPixelT kVal = *kernelIter;
            if (kVal != 0) {
                outPixel += static_cast<OutPixelT>((*imageIter) * kVal);
            }
        }
        return outPixel;
    }
    
    /*
     * Assert that the dimensions of convolvedImage, inImage and kernel are compatible with convolution.
     *
     * @throw lsst::pex::exceptions::InvalidParameterException if convolvedImage dimensions != inImage dim.
     * @throw lsst::pex::exceptions::InvalidParameterException if inImage smaller than kernel in width or h.
     * @throw lsst::pex::exceptions::InvalidParameterException if kernel width or height < 1
     */
    template <typename OutImageT, typename InImageT>
    void assertDimensionsOK(
        OutImageT const &convolvedImage,
        InImageT const &inImage,
        lsst::afw::math::Kernel const &kernel
    ) {
        if (convolvedImage.getDimensions() != inImage.getDimensions()) {
            std::ostringstream os;
            os << "convolvedImage dimensions = ( "
                << convolvedImage.getWidth() << ", " << convolvedImage.getHeight()
                << ") != (" << inImage.getWidth() << ", " << inImage.getHeight() << ") = inImage dimensions";
            throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
        }
        if (inImage.getDimensions() < kernel.getDimensions()) {
            std::ostringstream os;
            os << "inImage dimensions = ( "
                << inImage.getWidth() << ", " << inImage.getHeight()
                << ") smaller than (" << kernel.getWidth() << ", " << kernel.getHeight()
                << ") = kernel dimensions in width and/or height";
            throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
        }
        if ((kernel.getWidth() < 1) || (kernel.getHeight() < 1)) {
            std::ostringstream os;
            os << "kernel dimensions = ( "
                << kernel.getWidth() << ", " << kernel.getHeight()
                << ") smaller than (1, 1) in width and/or height";
            throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
        }
    }
    
}   // anonymous namespace


/**
 * Construct a SubregionIterator
 */
mathDetail::SubregionIterator::SubregionIterator(
        lsst::afw::geom::BoxI const &region,    ///< full region
        lsst::afw::geom::Extent2I const &subregionSize, ///< size of subregion (pixels)
        lsst::afw::geom::Extent2I const &overlapSize) ///< size of overlap (pixels)
:
    _region(region),
    _subregionSize(subregionSize),
    _overlapSize(overlapSize)
{
    if ((region.getWidth() < 1) || (region.getHeight() < 1)) {
        std::ostringstream os;
        os << "region size = ("
            << region.getWidth() << ", " << region.getHeight()
            << ") not positive in both dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    if ((subregionSize[0] < 1) || (subregionSize[1] < 1)) {
        std::ostringstream os;
        os << "subregionSize = ("
            << subregionSize[0] << ", " << subregionSize[1]
            << ") not positive in both dimensions";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
};

/**
 * Get the first subregion
 */
lsst::afw::geom::BoxI mathDetail::SubregionIterator::begin() const {
    afwGeom::BoxI retBox(_region.getMin(), _subregionSize);
    retBox.clip(_region);
    return retBox;
}

/**
 * Given a subregion, get the next subregion
 *
 * If the subregion is at the end of the region
 * then a subregion of size 0 one past the end is returned
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if subregion not fully contained in main region
 */
lsst::afw::geom::BoxI mathDetail::SubregionIterator::getNext(lsst::afw::geom::BoxI const &subregion) const {
    // sanity-check inputs
    if (!_region.contains(subregion)) {
        std::ostringstream os;
        os << "region = ( "
            << _region.getMinX() << ", " << _region.getMinY() << "; "
            << _region.getMaxX() << ", " << _region.getMaxY()
            << ") does not contain subregion ("
            << subregion.getMinX() << ", " << subregion.getMinY() << "; "
            << subregion.getMaxX() << ", " << subregion.getMaxY()
            << ")";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }

    int const maxX = _region.getMaxX();
    int const maxY = _region.getMaxY();
    int x0, y0;
    if (subregion.getMaxX() < maxX) {
        /// next box in this row (increment x and keep y the same)
        x0 = subregion.getMaxX() + 1 - _overlapSize.getX();
        y0 = subregion.getMinY();
    } else if (subregion.getMaxY() < maxY) {
        /// start next row (reset x to minimum and increment y)
        x0 = _region.getMinX();
        y0 = subregion.getMaxY() + 1 - _overlapSize.getY();
    } else {
        /// done
        return afwGeom::BoxI(afwGeom::makePointI(maxX + 1, maxY + 1), afwGeom::makeExtentI(0, 0));
    }
    int x1 = x0 + _subregionSize.getX() - 1;
    if (x1 > maxX) x1 = maxX;
    int y1 = y0 + _subregionSize.getY() - 1;
    if (y1 > maxY) y1 = maxY;
    return afwGeom::BoxI(afwGeom::makePointI(x0, y0), afwGeom::makePointI(x1, y1));
}

/**
 * @brief Low-level convolution function that does not set edge pixels.
 *
 * convolvedImage must be the same size as inImage.
 * convolvedImage has a border in which the output pixels are not set. This border has size:
 * - kernel.getCtrX() along the left edge
 * - kernel.getCtrY() along the bottom edge
 * - kernel.getWidth()  - 1 - kernel.getCtrX() along the right edge
 * - kernel.getHeight() - 1 - kernel.getCtrY() along the top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if convolvedImage dimensions != inImage dimensions
 * @throw lsst::pex::exceptions::InvalidParameterException if inImage smaller than kernel in width or height
 * @throw lsst::pex::exceptions::InvalidParameterException if kernel width or height < 1
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void mathDetail::basicConvolve(
        OutImageT &convolvedImage,      ///< convolved %image
        InImageT const& inImage,        ///< %image to convolve
        afwMath::Kernel const& kernel,  ///< convolution kernel
        afwMath::ConvolutionControl const& convolutionControl)  ///< convolution control parameters
{
    // Because convolve isn't a method of Kernel we can't always use Kernel's vtbl to dynamically
    // dispatch the correct version of basicConvolve. The case that fails is convolving with a kernel
    // obtained from a pointer or reference to a Kernel (base class), e.g. as used in LinearCombinationKernel.
    if (ISINSTANCE(kernel, afwMath::DeltaFunctionKernel)) {
        pexLog::TTrace<4>("lsst.afw.math.convolve",
            "generic basicConvolve: dispatch to DeltaFunctionKernel basicConvolve");
        mathDetail::basicConvolve(convolvedImage, inImage,
            *dynamic_cast<afwMath::DeltaFunctionKernel const*>(&kernel),
            convolutionControl);
        return;
    } else if (ISINSTANCE(kernel, afwMath::SeparableKernel)) {
        pexLog::TTrace<4>("lsst.afw.math.convolve",
            "generic basicConvolve: dispatch to SeparableKernel basicConvolve");
        mathDetail::basicConvolve(convolvedImage, inImage,
            *dynamic_cast<afwMath::SeparableKernel const*>(&kernel),
            convolutionControl);
        return;
    } else if (ISINSTANCE(kernel, afwMath::LinearCombinationKernel) && kernel.isSpatiallyVarying()) {
        pexLog::TTrace<4>("lsst.afw.math.convolve",
            "generic basicConvolve: dispatch to spatially varying LinearCombinationKernel basicConvolve");
        mathDetail::basicConvolve(convolvedImage, inImage,
            *dynamic_cast<afwMath::LinearCombinationKernel const*>(&kernel),
            convolutionControl);
        return;
    }
    // OK, use general (and slower) form
    if (kernel.isSpatiallyVarying() && (convolutionControl.getMaxInterpolationError() > 0.0)) {
        // use linear interpolation
        pexLog::TTrace<3>("lsst.afw.math.convolve", "generic basicConvolve: using linear interpolation");
        mathDetail::convolveWithInterpolation(convolvedImage, inImage, kernel, convolutionControl);

    } else {
        // use brute force
        pexLog::TTrace<3>("lsst.afw.math.convolve", "generic basicConvolve: using brute force");
        mathDetail::convolveWithBruteForce(convolvedImage, inImage, kernel,
            convolutionControl.getDoNormalize());
    }
}

/**
 * @brief A version of basicConvolve that should be used when convolving delta function kernels
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void mathDetail::basicConvolve(
        OutImageT& convolvedImage,      ///< convolved %image
        InImageT const& inImage,        ///< %image to convolve
        afwMath::DeltaFunctionKernel const &kernel, ///< convolution kernel
        afwMath::ConvolutionControl const &)        ///< unused
{
    assert (!kernel.isSpatiallyVarying());
    assertDimensionsOK(convolvedImage, inImage, kernel);
    
    int const mImageWidth = inImage.getWidth(); // size of input region
    int const mImageHeight = inImage.getHeight();
    int const cnvWidth = mImageWidth + 1 - kernel.getWidth();
    int const cnvHeight = mImageHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();
    int const inStartX = kernel.getPixel().getX();
    int const inStartY = kernel.getPixel().getY();

    pexLog::TTrace<3>("lsst.afw.math.convolve", "DeltaFunctionKernel basicConvolve");

    for (int i = 0; i < cnvHeight; ++i) {
        typename InImageT::x_iterator inPtr = inImage.x_at(inStartX, i +  inStartY);
        for (typename OutImageT::x_iterator cnvPtr = convolvedImage.x_at(cnvStartX, i + cnvStartY),
                 cnvEnd = cnvPtr + cnvWidth; cnvPtr != cnvEnd; ++cnvPtr, ++inPtr){
            *cnvPtr = *inPtr;
        }
    }
}

/**
 * @brief A version of basicConvolve that should be used when convolving a LinearCombinationKernel
 *
 * The Algorithm:
 * - If the kernel is spatially varying, then convolves the input Image by each basis kernel in turn,
 *   solves the spatial model for that component and adds in the appropriate amount of the convolved %image.
 * - If the kernel is not spatially varying, then computes a fixed kernel and calls the
 *   the general version of basicConvolve.
 *
 * @warning The variance will be mis-computed if your basis kernels contain home-brew delta function kernels
 * (instead of instances of afwMath::DeltaFunctionKernel). It may also be mis-computed if your basis kernels
 * contain many pixels with value zero, or if your basis kernels contain a mix of
 * afwMath::DeltaFunctionKernel with other kernels.
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if convolvedImage dimensions != inImage dimensions
 * @throw lsst::pex::exceptions::InvalidParameterException if inImage smaller than kernel in width or height
 * @throw lsst::pex::exceptions::InvalidParameterException if kernel width or height < 1
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void mathDetail::basicConvolve(
    OutImageT& convolvedImage,      ///< convolved %image
    InImageT const& inImage,        ///< %image to convolve
    afwMath::LinearCombinationKernel const& kernel,         ///< convolution kernel
    afwMath::ConvolutionControl const & convolutionControl) ///< convolution control parameters
{
    if (!kernel.isSpatiallyVarying()) {
        // use the standard algorithm for the spatially invariant case
        pexLog::TTrace<3>("lsst.afw.math.convolve",
            "basicConvolve for LinearCombinationKernel: spatially invariant; using brute force");
        return mathDetail::convolveWithBruteForce(convolvedImage, inImage, kernel,
            convolutionControl.getDoNormalize());
    } else if (!kernel.isDeltaFunctionBasis()) {
        // use the standard algorithm for the spatially varying case
        if (convolutionControl.getMaxInterpolationError() > 0.0) {
            pexLog::TTrace<3>("lsst.afw.math.convolve",
                "basicConvolve for LinearCombinationKernel: using interpolation");
            return mathDetail::convolveWithInterpolation(convolvedImage, inImage, kernel, convolutionControl);
        } else {
            pexLog::TTrace<3>("lsst.afw.math.convolve",
                "basicConvolve for LinearCombinationKernel: maxInterpolationError < 0; using brute force");
            return mathDetail::convolveWithBruteForce(convolvedImage, inImage, kernel,
                convolutionControl.getDoNormalize());
        }
    }
    
    // use specialization for delta function basis; this is faster then
    // the standard algorithm but requires more memory

    assertDimensionsOK(convolvedImage, inImage, kernel);
    
    pexLog::TTrace<3>("lsst.afw.math.convolve",
        "basicConvolve for LinearCombinationKernel: spatially varying delta function basis");
    
    typedef typename InImageT::template ImageTypeFactory<double>::type BasisImage;
    typedef typename BasisImage::x_iterator BasisXIterator;
    typedef typename OutImageT::x_iterator OutXIterator;
    typedef afwMath::KernelList KernelList;

    int const imWidth = inImage.getWidth();
    int const imHeight = inImage.getHeight();
    int const cnvWidth = imWidth + 1 - kernel.getWidth();
    int const cnvHeight = imHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();
    int const cnvEndX = cnvStartX + cnvWidth;  // end index + 1
    int const cnvEndY = cnvStartY + cnvHeight; // end index + 1
    // create a BasisImage to hold the source convolved with a basis kernel
    BasisImage basisImage(inImage.getDimensions());

    // initialize good area of output image to zero so we can add the convolved basis images into it
    typename OutImageT::SinglePixel const nullPixel(0);
    for (int cnvY = cnvStartY; cnvY < cnvEndY; ++cnvY) {
        OutXIterator cnvXIter = convolvedImage.row_begin(cnvY) + cnvStartX;
        for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++cnvXIter) {
            *cnvXIter = nullPixel;
        }
    }
    
    // iterate over basis kernels
    KernelList const basisKernelList = kernel.getKernelList();
    int i = 0;
    for (typename KernelList::const_iterator basisKernelIter = basisKernelList.begin();
        basisKernelIter != basisKernelList.end(); ++basisKernelIter, ++i) {
        mathDetail::basicConvolve(basisImage, inImage, **basisKernelIter, false);

        // iterate over matching pixels of all images to compute output image
        afwMath::Kernel::SpatialFunctionPtr spatialFunctionPtr = kernel.getSpatialFunction(i);
        std::vector<double> kernelCoeffList(kernel.getNKernelParameters());
            // weights of basis images at this point
        for (int cnvY = cnvStartY; cnvY < cnvEndY; ++cnvY) {
            double const rowPos = inImage.indexToPosition(cnvY, afwImage::Y);
        
            OutXIterator cnvXIter = convolvedImage.row_begin(cnvY) + cnvStartX;
            BasisXIterator basisXIter = basisImage.row_begin(cnvY) + cnvStartX;
            for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++cnvXIter, ++basisXIter) {
                double const colPos = inImage.indexToPosition(cnvX, afwImage::X);
                double basisCoeff = (*spatialFunctionPtr)(colPos, rowPos);
                
                typename OutImageT::SinglePixel cnvPixel(*cnvXIter);
                cnvPixel += (*basisXIter) * basisCoeff;
                *cnvXIter = cnvPixel;
                // note: cnvPixel avoids compiler complaints; the following does not build:
                // *cnvXIter += (*basisXIter) * basisCoeff;
            }
        }
    }

    if (convolutionControl.getDoNormalize()) {
        /*
        For each pixel of the output image: compute the kernel sum for that pixel and scale
        the output image. One obvious alternative is to create a temporary kernel sum image
        and accumulate into that while iterating over the basis kernels above. This saves
        computing the spatial functions again here, but requires a temporary image
        the same size as the output image, so it is likely to suffer from cache issues.
        */
        std::vector<double> const kernelSumList = kernel.getKernelSumList();
        std::vector<Kernel::SpatialFunctionPtr> spatialFunctionList = kernel.getSpatialFunctionList();
        for (int cnvY = cnvStartY; cnvY < cnvEndY; ++cnvY) {
            double const rowPos = inImage.indexToPosition(cnvY, afwImage::Y);
        
            OutXIterator cnvXIter = convolvedImage.row_begin(cnvY) + cnvStartX;
            for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++cnvXIter) {
                double const colPos = inImage.indexToPosition(cnvX, afwImage::X);

                std::vector<double>::const_iterator kSumIter = kernelSumList.begin();
                std::vector<Kernel::SpatialFunctionPtr>::const_iterator spFuncIter =
                    spatialFunctionList.begin();
                double kSum = 0.0;
                for ( ; kSumIter != kernelSumList.end(); ++kSumIter, ++spFuncIter) {
                    kSum += (**spFuncIter)(colPos, rowPos) * (*kSumIter);
                }
                *cnvXIter /= kSum;
            }
        }
    }
}

/**
 * @brief A version of basicConvolve that should be used when convolving separable kernels
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void mathDetail::basicConvolve(
        OutImageT& convolvedImage,      ///< convolved %image
        InImageT const& inImage,        ///< %image to convolve
        afwMath::SeparableKernel const &kernel, ///< convolution kernel
        afwMath::ConvolutionControl const & convolutionControl) ///< convolution control parameters
{
    typedef typename afwMath::Kernel::Pixel KernelPixel;
    typedef typename std::vector<KernelPixel> KernelVector;
    typedef KernelVector::const_iterator KernelIterator;
    typedef typename InImageT::const_x_iterator InXIterator;
    typedef typename InImageT::const_xy_locator InXYLocator;
    typedef typename OutImageT::x_iterator OutXIterator;
    typedef typename OutImageT::y_iterator OutYIterator;
    typedef typename OutImageT::SinglePixel OutPixel;

    assertDimensionsOK(convolvedImage, inImage, kernel);
    
    int const imWidth = inImage.getWidth();
    int const imHeight = inImage.getHeight();
    int const kWidth = kernel.getWidth();
    int const kHeight = kernel.getHeight();
    int const cnvWidth = static_cast<int>(imWidth) + 1 - static_cast<int>(kernel.getWidth());
    int const cnvHeight = static_cast<int>(imHeight) + 1 - static_cast<int>(kernel.getHeight());
    int const cnvStartX = static_cast<int>(kernel.getCtrX());
    int const cnvStartY = static_cast<int>(kernel.getCtrY());
    int const cnvEndX = cnvStartX + cnvWidth; // end index + 1
    int const cnvEndY = cnvStartY + cnvHeight; // end index + 1

    KernelVector kXVec(kWidth);
    KernelVector kYVec(kHeight);
    
    if (kernel.isSpatiallyVarying()) {
        pexLog::TTrace<3>("lsst.afw.math.convolve",
            "SeparableKernel basicConvolve: kernel is spatially varying");

        for (int cnvY = cnvStartY; cnvY != cnvEndY; ++cnvY) {
            double const rowPos = inImage.indexToPosition(cnvY, afwImage::Y);
            
            InXYLocator  inImLoc =  inImage.xy_at(0, cnvY - cnvStartY);
            OutXIterator cnvXIter = convolvedImage.row_begin(cnvY) + cnvStartX;
            for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++inImLoc.x(), ++cnvXIter) {
                double const colPos = inImage.indexToPosition(cnvX, afwImage::X);

                KernelPixel kSum = kernel.computeVectors(kXVec, kYVec, convolutionControl.getDoNormalize(),
                    colPos, rowPos);

                // why does this trigger warnings? It did not in the past.
                *cnvXIter = afwMath::convolveAtAPoint<OutImageT, InImageT>(inImLoc, kXVec, kYVec);
                if (convolutionControl.getDoNormalize()) {
                    *cnvXIter = *cnvXIter/kSum;
                }
            }
        }
    } else {
        // kernel is spatially invariant
        pexLog::TTrace<3>("lsst.afw.math.convolve",
            "SeparableKernel basicConvolve: kernel is spatially invariant");

        kernel.computeVectors(kXVec, kYVec, convolutionControl.getDoNormalize());
        KernelIterator const kXVecBegin = kXVec.begin();
        KernelIterator const kYVecBegin = kYVec.begin();

        // Handle the x kernel vector first, putting results into convolved image as a temporary buffer
        // (all remaining processing must read from and write to the convolved image,
        // thus being careful not to modify pixels that still need to be read)
        for (int imageY = 0; imageY < imHeight; ++imageY) {
            OutXIterator cnvXIter = convolvedImage.x_at(cnvStartX, imageY);
            InXIterator inXIter = inImage.x_at(0, imageY);
            InXIterator const inXIterEnd = inImage.x_at(cnvWidth, imageY);
            for ( ; inXIter != inXIterEnd; ++cnvXIter, ++inXIter) {
                *cnvXIter = kernelDotProduct<OutPixel, InXIterator, KernelIterator, KernelPixel>(
                    inXIter, kXVecBegin, kWidth);
            }
        }
        
        // Handle the y kernel vector. It turns out to be faster for the innermost loop to be along y,
        // probably because one can accumulate into a temporary variable.
        // For each row of output, compute the output pixel, putting it at the bottom
        // (a pixel that will not be read again).
        // The resulting image is correct, but shifted down by kernel ctr y pixels.
        for (int cnvY = 0; cnvY < cnvHeight; ++cnvY) {
            for (int x = cnvStartX; x < cnvEndX; ++x) {
                OutYIterator cnvYIter = convolvedImage.y_at(x, cnvY);
                *cnvYIter = kernelDotProduct<OutPixel, OutYIterator, KernelIterator, KernelPixel>(
                    cnvYIter, kYVecBegin, kHeight);
            }
        }

        // Move the good pixels up by kernel ctr Y (working down to avoid overwriting data)
        for (int destY = cnvEndY - 1, srcY = cnvHeight - 1; srcY >= 0; --destY, --srcY) {
            OutXIterator destIter = convolvedImage.x_at(cnvStartX, destY);
            OutXIterator const destIterEnd = convolvedImage.x_at(cnvEndX, destY);
            OutXIterator srcIter = convolvedImage.x_at(cnvStartX, srcY);
            for ( ; destIter != destIterEnd; ++destIter, ++srcIter) {
                *destIter = *srcIter;
            }
        }
    }
}

/**
 * @brief Convolve an Image or MaskedImage with a Kernel by computing the kernel image
 * at every point. (If the kernel is not spatially varying then only compute it once).
 *
 * @warning Low-level convolution function that does not set edge pixels.
 *
 * convolvedImage must be the same size as inImage.
 * convolvedImage has a border in which the output pixels are not set. This border has size:
 * - kernel.getCtrX() along the left edge
 * - kernel.getCtrY() along the bottom edge
 * - kernel.getWidth()  - 1 - kernel.getCtrX() along the right edge
 * - kernel.getHeight() - 1 - kernel.getCtrY() along the top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if convolvedImage dimensions != inImage dimensions
 * @throw lsst::pex::exceptions::InvalidParameterException if inImage smaller than kernel in width or height
 * @throw lsst::pex::exceptions::InvalidParameterException if kernel width or height < 1
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void mathDetail::convolveWithBruteForce(
        OutImageT &convolvedImage,      ///< convolved %image
        InImageT const& inImage,        ///< %image to convolve
        afwMath::Kernel const& kernel,  ///< convolution kernel
        bool doNormalize)               ///< if true, normalize the kernel, else use "as is"
{
    typedef typename afwMath::Kernel::Pixel KernelPixel;
    typedef afwImage::Image<KernelPixel> KernelImage;

    typedef typename KernelImage::const_x_iterator KernelXIterator;
    typedef typename KernelImage::const_xy_locator KernelXYLocator;
    typedef typename InImageT::const_x_iterator InXIterator;
    typedef typename InImageT::const_xy_locator InXYLocator;
    typedef typename OutImageT::x_iterator OutXIterator;
    typedef typename OutImageT::SinglePixel OutPixel;

    assertDimensionsOK(convolvedImage, inImage, kernel);
    
    int const inImageWidth = inImage.getWidth();
    int const inImageHeight = inImage.getHeight();
    int const kWidth = kernel.getWidth();
    int const kHeight = kernel.getHeight();
    int const cnvWidth = inImageWidth + 1 - kernel.getWidth();
    int const cnvHeight = inImageHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();
    int const cnvEndX = cnvStartX + cnvWidth;  // end index + 1
    int const cnvEndY = cnvStartY + cnvHeight; // end index + 1

    KernelImage kernelImage(kernel.getDimensions());
    KernelXYLocator const kernelLoc = kernelImage.xy_at(0,0);

    if (kernel.isSpatiallyVarying()) {
        pexLog::TTrace<5>("lsst.afw.math.convolve",
            "convolveWithBruteForce: kernel is spatially varying");

        for (int cnvY = cnvStartY; cnvY != cnvEndY; ++cnvY) {
            double const rowPos = inImage.indexToPosition(cnvY, afwImage::Y);
            
            InXYLocator  inImLoc =  inImage.xy_at(0, cnvY - cnvStartY);
            OutXIterator cnvXIter = convolvedImage.x_at(cnvStartX, cnvY);
            for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++inImLoc.x(), ++cnvXIter) {
                double const colPos = inImage.indexToPosition(cnvX, afwImage::X);

                KernelPixel kSum = kernel.computeImage(kernelImage, false, colPos, rowPos);
                *cnvXIter = afwMath::convolveAtAPoint<OutImageT, InImageT>(
                    inImLoc, kernelLoc, kWidth, kHeight);
                if (doNormalize) {
                    *cnvXIter = *cnvXIter/kSum;
                }
            }
        }
    } else {
        pexLog::TTrace<5>("lsst.afw.math.convolve",
            "convolveWithBruteForce: kernel is spatially invariant");
        (void)kernel.computeImage(kernelImage, doNormalize);
        
        for (int inStartY = 0, cnvY = cnvStartY; inStartY < cnvHeight; ++inStartY, ++cnvY) {
            for (OutXIterator cnvXIter=convolvedImage.x_at(cnvStartX, cnvY),
                cnvXEnd = convolvedImage.row_end(cnvY); cnvXIter != cnvXEnd; ++cnvXIter) {
                *cnvXIter = 0;
            }
            for (int kernelY = 0, inY = inStartY; kernelY < kHeight; ++inY, ++kernelY) {
                KernelXIterator kernelXIter = kernelImage.x_at(0, kernelY);
                InXIterator inXIter = inImage.x_at(0, inY);
                OutXIterator cnvXIter = convolvedImage.x_at(cnvStartX, cnvY);
                for (int x = 0; x < cnvWidth; ++x, ++cnvXIter, ++inXIter) {
                    *cnvXIter += kernelDotProduct<OutPixel, InXIterator, KernelXIterator, KernelPixel>(
                        inXIter, kernelXIter, kWidth);
                }
            }
        }
    }
}

/*
 * Explicit instantiation
 *
 * Modelled on ConvolveImage.cc
 */
#define IMAGE(PIXTYPE) afwImage::Image<PIXTYPE>
#define MASKEDIMAGE(PIXTYPE) afwImage::MaskedImage<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
#define NL /* */
// Instantiate Image or MaskedImage versions
#define INSTANTIATE_IM_OR_MI(IMGMACRO, OUTPIXTYPE, INPIXTYPE) \
    template void mathDetail::basicConvolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::Kernel const&, \
            afwMath::ConvolutionControl const&); NL \
    template void mathDetail::basicConvolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::DeltaFunctionKernel const&, \
            afwMath::ConvolutionControl const&); NL \
    template void mathDetail::basicConvolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::LinearCombinationKernel const&, \
            afwMath::ConvolutionControl const&); NL \
    template void mathDetail::basicConvolve( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::SeparableKernel const&, \
            afwMath::ConvolutionControl const&); NL \
    template void mathDetail::convolveWithBruteForce( \
        IMGMACRO(OUTPIXTYPE)&, IMGMACRO(INPIXTYPE) const&, afwMath::Kernel const&, bool);
// Instantiate both Image and MaskedImage versions
#define INSTANTIATE(OUTPIXTYPE, INPIXTYPE) \
    INSTANTIATE_IM_OR_MI(IMAGE,       OUTPIXTYPE, INPIXTYPE) \
    INSTANTIATE_IM_OR_MI(MASKEDIMAGE, OUTPIXTYPE, INPIXTYPE)

INSTANTIATE(double, double)
INSTANTIATE(double, float)
INSTANTIATE(double, int)
INSTANTIATE(double, boost::uint16_t)
INSTANTIATE(float, float)
INSTANTIATE(float, int)
INSTANTIATE(float, boost::uint16_t)
INSTANTIATE(int, int)
INSTANTIATE(boost::uint16_t, boost::uint16_t)
