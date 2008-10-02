// -*- LSST-C++ -*-
/**
 * @file
 *
* @brief Definition of functions declared in ConvolveMaskedImage.h
 *
 * This file is meant to be included by lsst/afw/math/ConvolveMaskedImage.h
 *
 * @todo
 * * Speed up convolution
 *
 * @note: the convolution and apply functions assume that data in a row is contiguous,
 * both in the input image and in the kernel. This will eventually be enforced by afw.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>
#include <string>

#include "boost/format.hpp"

#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image/ImageUtils.h"

// if defined then kernel pixels that have value 0 are ignored when computing the output mask during convolution
#define IgnoreKernelZeroPixels

// Private functions to copy Images' borders
#include "lsst/afw/math/copyEdges.h"

/**
 * @brief Apply convolution kernel to a masked image at one point
 *
 * Note: this is a high performance routine; the user is expected to:
 * - handle edge extension
 * - figure out the kernel center and adjust the supplied pixel accessors accordingly
 * For an example of how to do this see the convolve function.
 *
 * @ingroup afw
 */
template <typename OutMaskedImageT, typename InMaskedImageT>
inline void lsst::afw::math::apply(
#if 0
    typename OutMaskedImageT::xy_locator& outLocator,
					///< locator for output pixel
#else
    typename OutMaskedImageT::IMV_tuple const& outLocator, // triple of (image, mask, variance)
#endif
    typename InMaskedImageT::const_xy_locator& inLocator,
                                        ///< locator for masked image pixel that overlaps (0,0) pixel of kernel(!)
    typename lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::const_xy_locator& kernelLocator,
                                        ///< accessor for (0,0) pixel of kernel
    int kwidth,                         ///< number of columns in kernel
    int kheight                         ///< number of rows in kernel
) {
    double outImage = 0;
    double outVariance = 0;
    long outMask = 0;

    for (int y = 0; y != kheight; ++y) {
        for (int x = 0; x != kwidth; ++x, ++inLocator.x(), ++kernelLocator.x()) {
            lsst::afw::math::Kernel::PixelT ker = kernelLocator[0];

#ifdef IgnoreKernelZeroPixels
            if (ker != 0)
#endif
            {
                outImage += ker*inLocator.image();
                outMask |= inLocator.mask();
                outVariance += ker*ker*inLocator.variance();
            }
            
        }

        inLocator  += lsst::afw::image::details::difference_type(-kwidth, 1);
        kernelLocator += lsst::afw::image::details::difference_type(-kwidth, 1);
    }

#if 0
    outLocator.image() = outImage;
    outLocator.mask() = outMask;
    outLocator.variance() = outVariance;
#else
    outLocator.template get<0>() = outImage;
    outLocator.template get<1>() = outMask;
    outLocator.template get<2>() = outVariance;
#endif
}

/**
 * @brief Apply separable convolution kernel to a masked image at one point
 *
 * Note: this is a high performance routine; the user is expected to:
 * - handle edge extension
 * - figure out the kernel center and adjust the supplied pixel accessors accordingly
 * For an example of how to do this see the convolve function.
 *
 * @ingroup afw
 */
template <typename InMaskedImageT, typename OutMaskedImageT>
inline void lsst::afw::math::apply(
        typename OutMaskedImageT::xy_locator& outLocator,
					///< locator for output pixel
        typename InMaskedImageT::const_xy_locator& inLocator,
                                        ///< locator for masked image pixel that overlaps (0,0) pixel of kernel(!)
        std::vector<lsst::afw::math::Kernel::PixelT> const &kernelXList,  ///< kernel column vector
        std::vector<lsst::afw::math::Kernel::PixelT> const &kernelYList   ///< kernel row vector
) {
    typedef typename std::vector<lsst::afw::math::Kernel::PixelT>::const_iterator k_iter;

    double outImage = 0;
    double outVariance = 0;
    long outMask = 0;
    for (k_iter kernelYIter = kernelYList.begin(), end = kernelYList.end();
         kernelYIter != end; ++kernelYIter) {

        double outImageY = 0;
        double outVarianceY = 0;
        long outMaskY = 0;

        for (k_iter kernelXIter = kernelXList.begin(), end = kernelXList.end();
             kernelXIter != end; ++kernelXIter, ++inLocator.x()) {
            double kernelXValue = *kernelXIter;
#ifdef IgnoreKernelZeroPixels
            if (kernelXValue != 0)
#endif
            {
                outImageY += kernelXValue*inLocator.image();
                outMaskY |= inLocator.mask();
                outVarianceY += kernelXValue*kernelXValue*inLocator.variance();
            }
        }

        double kernelYValue = *kernelYIter;
#ifdef IgnoreKernelZeroPixels
        if (kernelYValue != 0)
#endif
        {
            outImage +=    outImageY*kernelYValue;
            outMask |=     outMaskY;
            outVariance += outVarianceY*kernelYValue*kernelYValue;
        }

        inLocator += lsst::afw::image::details::difference_type(-kernelXList.size(), 1);
    }
    
    outLocator.image() = outImage;
    outLocator.mask() = outMask;
    outLocator.variance() = outVariance;
}

#if 0                                   // The template in ConvolveImage.cc is the same as this one
/**
 * @brief Low-level convolution function that does not set edge pixels.
 *
 * convolvedImage must be the same size as maskedImage.
 * convolvedImage has a border in which the output pixels are not set. This border has size:
 * * kernel.getCtrX/Y() along the left/bottom edge
 * * kernel.getWidth/Height() - 1 - kernel.getCtrX/Y() along the right/top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameter if convolvedImage is not the same size as maskedImage.
 * @throw lsst::pex::exceptions::InvalidParameter if maskedImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename OutMaskedImageT, typename InMaskedImageT>
void lsst::afw::math::basicConvolve(
    OutMaskedImageT& convolvedImage,            ///< convolved image
    InMaskedImageT const& inImage,              ///< image to convolve
    lsst::afw::math::Kernel const &kernel,      ///< convolution kernel
    bool doNormalize                            ///< if True, normalize the kernel, else use "as is"
) {
    typedef typename lsst::afw::math::Kernel::PixelT KernelPixelT;

    typedef typename lsst::afw::image::Image<KernelPixelT>::const_xy_locator kXY_locator;
    typedef typename InMaskedImageT::const_xy_locator inXY_locator;
    typedef typename OutMaskedImageT::x_iterator cnvX_iterator;

    if (convolvedImage.dimensions() != inImage.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if (inImage.dimensions() < kernel.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("inImage smaller than kernel in columns and/or rows");
    }
    
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

    if (kernel.isSpatiallyVarying()) {
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is spatially varying");

        lsst::afw::image::Image<KernelPixelT> kernelImage(kernel.dimensions()); // the kernel at a point

        for (int cnvY = cnvStartY; cnvY != cnvEndY; ++cnvY) {
            double const rowPos = lsst::afw::image::indexToPosition(cnvY);
            
            inXY_locator  inImLoc =  inImage.xy_at(0, cnvY - cnvStartY);
            cnvX_iterator cnvImIter = convolvedImage.row_begin(cnvY) + cnvStartX;
            for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++inImLoc.x(), ++cnvImIter) {
                double const colPos = lsst::afw::image::indexToPosition(cnvX);

                KernelPixelT kSum;
                kernel.computeImage(kernelImage, kSum, false, colPos, rowPos);

                kXY_locator kernelLoc = kernelImage.xy_at(0,0);
                lsst::afw::math::apply(*cnvImIter, inImLoc, kernelLoc, kWidth, kHeight);
                if (doNormalize) {
                    *cnvImIter /= kSum;
                }
            }
        }
    } else {                            // kernel is spatially invariant
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is spatially invariant");
        KernelPixelT kSum;
        lsst::afw::image::Image<KernelPixelT> kernelImage = kernel.computeNewImage(kSum, doNormalize);

        for (int cnvY = cnvStartY; cnvY != cnvEndY; ++cnvY) {
            inXY_locator inImLoc =  inImage.xy_at(0, cnvY - cnvStartY);
            for (cnvX_iterator cnvImIter = convolvedImage.row_begin(cnvY) + cnvStartX,
                     cnvImEnd = cnvImIter + cnvEndX - cnvStartX; cnvImIter != cnvImEnd; ++inImLoc.x(), ++cnvImIter) {
                kXY_locator kernelLoc = kernelImage.xy_at(0,0);
                lsst::afw::math::apply(*cnvImIter, inImLoc, kernelLoc, kWidth, kHeight);
            }
        }
    }
}
#endif

/************************************************************************************************************/
/**
 * @brief A version of basicConvolve that should be used when convolving delta function kernels
 */
template <typename OutMaskedImageT, typename InMaskedImageT>
void lsst::afw::math::basicConvolve(
    OutMaskedImageT& convolvedImage,            ///< convolved image
    InMaskedImageT const& inImage,          ///< image to convolve
    lsst::afw::math::DeltaFunctionKernel const &kernel,    ///< convolution kernel
    bool doNormalize    ///< if True, normalize the kernel, else use "as is"
) {
    assert (!kernel.isSpatiallyVarying());

    if (convolvedImage.dimensions() != inImage.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if (convolvedImage.dimensions() < kernel.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter(
            "inImage smaller than kernel in columns and/or rows");
    }
    
    int const mImageWidth = inImage.getWidth(); // size of input region
    int const mImageHeight = inImage.getHeight();
    int const cnvWidth = mImageWidth + 1 - kernel.getWidth();
    int const cnvHeight = mImageHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();
    int const inStartX = kernel.getPixel().first;
    int const inStartY = kernel.getPixel().second;

    lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially invariant delta function basis");

    for (int i = 0; i < cnvHeight; ++i) {
        typename InMaskedImageT::x_iterator
            inPtr =         inImage.row_begin(i +  inStartY) +  inStartX;
        typename OutMaskedImageT::x_iterator
            cnvPtr = convolvedImage.row_begin(i + cnvStartY) + cnvStartX,
            cnvEnd = cnvPtr + cnvWidth;

        for (; cnvPtr != cnvEnd; ++cnvPtr, ++inPtr) {
            cnvPtr.image() = inPtr.image();
            cnvPtr.mask() = inPtr.mask();
            cnvPtr.variance() = inPtr.variance();
        }
    }
}

/**
 * @brief A version of basicConvolve that should be used when convolving separable kernels
 */
template <typename OutMaskedImageT, typename InMaskedImageT>
void lsst::afw::math::basicConvolve(
    OutMaskedImageT& convolvedImage,        ///< convolved image
    InMaskedImageT const& inImage,          ///< image to convolve
    lsst::afw::math::SeparableKernel const &kernel,  ///< convolution kernel
    bool doNormalize                                 ///< if True, normalize the kernel, else use "as is"
) {
    typedef typename lsst::afw::math::Kernel::PixelT KernelPixelT;

    typedef typename InMaskedImageT::const_xy_locator inXY_locator;
    typedef typename OutMaskedImageT::x_iterator cnvX_iterator;

    if (convolvedImage.dimensions() != inImage.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if (inImage.dimensions() < kernel.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter(
            "inImage smaller than kernel in columns and/or rows");
    }
    
    int const imWidth = inImage.getWidth();
    int const imHeight = inImage.getHeight();
    int const cnvWidth = static_cast<int>(imWidth) + 1 - static_cast<int>(kernel.getWidth());
    int const cnvHeight = static_cast<int>(imHeight) + 1 - static_cast<int>(kernel.getHeight());
    int const cnvStartX = static_cast<int>(kernel.getCtrX());
    int const cnvStartY = static_cast<int>(kernel.getCtrY());
    int const cnvEndX = cnvStartX + static_cast<int>(cnvWidth); // end index + 1
    int const cnvEndY = cnvStartY + static_cast<int>(cnvHeight); // end index + 1

    std::vector<lsst::afw::math::Kernel::PixelT> kXVec(kernel.getWidth());
    std::vector<lsst::afw::math::Kernel::PixelT> kYVec(kernel.getHeight());
    
    if (kernel.isSpatiallyVarying()) {
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially varying separable kernel");

        for (int cnvY = cnvStartY; cnvY < cnvEndY; ++cnvY) {
            double const rowPos = lsst::afw::image::indexToPosition(cnvY);
            
            inXY_locator  inImLoc =  inImage.xy_at(0, cnvY - cnvStartY);
            cnvX_iterator cnvImIter = convolvedImage.row_begin(cnvY) + cnvStartX;
            for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++inImLoc.x(), ++cnvImIter) {
                double const colPos = lsst::afw::image::indexToPosition(cnvX);

                KernelPixelT kSum;
                kernel.computeVectors(kXVec, kYVec, kSum, doNormalize, colPos, rowPos);

                lsst::afw::math::apply(*cnvImIter, inImLoc, kXVec, kYVec);
                if (doNormalize) {
                    *cnvImIter /= kSum;
                }
            }
        }
    } else {
        // kernel is spatially invariant
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially invariant separable kernel");

        KernelPixelT kSum;
        kernel.computeVectors(kXVec, kYVec, kSum, doNormalize);

        for (int cnvY = cnvStartY; cnvY != cnvEndY; ++cnvY) {
            inXY_locator inImLoc =  inImage.xy_at(0, cnvY - cnvStartY);
            for (cnvX_iterator cnvImIter = convolvedImage.row_begin(cnvY) + cnvStartX,
                     cnvImEnd = cnvImIter + cnvEndX - cnvStartX; cnvImIter != cnvImEnd; ++inImLoc.x(), ++cnvImIter) {
                lsst::afw::math::apply(*cnvImIter, inImLoc, kXVec, kYVec);
            }
        }
    }
}

/**
 * @brief Convolve a MaskedImage with a Kernel, setting pixels of an existing image
 *
 * convolvedImage must be the same size as inImage.
 * convolvedImage has a border in which the output pixels are just a copy of the input pixels
 * and the output mask bit edgeBit has been set. This border has size:
 * * kernel.getCtrX/Y() along the left/bottom edge
 * * kernel.getWidth/Height() - 1 - kernel.getCtrX/Y() along the right/top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameter if convolvedImage is not the same size as inImage.
 * @throw lsst::pex::exceptions::InvalidParameter if inImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT, typename KernelT>
void lsst::afw::math::convolve(
    OutImageT& convolvedImage,          ///< convolved image
    InImageT const& inImage,            ///< image to convolve
    KernelT const &kernel,              ///< convolution kernel
    int edgeBit,                        ///< mask bit to indicate pixel includes edge-extended data;
                                        ///< if negative then no bit is set
    bool doNormalize                    ///< if True, normalize the kernel, else use "as is"
) {
    lsst::afw::math::basicConvolve(convolvedImage, inImage, kernel, doNormalize);
    _copyBorder(convolvedImage, inImage, kernel, edgeBit);
}

/**
 * @brief Convolve a MaskedImage with a Kernel, returning a new image.
 *
 * @return the convolved MaskedImage.
 *
 * The returned MaskedImage is the same size as inImage.
 * It has a border in which the output pixels are just a copy of the input pixels
 * and the output mask bit edgeBit has been set. This border will have size:
 * * kernel.getCtrX/Y() along the left/bottom edge
 * * kernel.getWidth/Height() - 1 - kernel.getCtrX/Y() along the right/top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameter if inImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename ImageT, typename KernelT>
ImageT lsst::afw::math::convolveNew(
    ImageT const &inImage,              ///< image to convolve
    KernelT const &kernel,              ///< convolution kernel
    int edgeBit,                        ///< mask bit to indicate pixel includes edge-extended data;
                                        ///< if negative then no bit is set
    bool doNormalize                    ///< if True, normalize the kernel, else use "as is"
                                   ) {
    ImageT convolvedImage(inImage.dimensions());
    lsst::afw::math::convolve(convolvedImage, inImage, kernel, edgeBit, doNormalize);
    return convolvedImage;
}

/**
 * @brief Convolve a MaskedImage with a LinearCombinationKernel, setting pixels of an existing image.
 *
 * A variant of the convolve function that is faster for spatially varying LinearCombinationKernels.
 * For the sake of speed the kernel is NOT normalized. If you want normalization then call the standard
 * convolve function.
 *
 * The Algorithm:
 * Convolves the input MaskedImage by each basis kernel in turn, creating a set of basis images.
 * Then for each output pixel it solves the spatial model and computes the the pixel as
 * the appropriate linear combination of basis images.
 *
 * @throw lsst::pex::exceptions::InvalidParameter if convolvedImage is not the same size as inImage.
 * @throw lsst::pex::exceptions::InvalidParameter if inImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename OutMaskedImageT, typename InMaskedImageT>
void lsst::afw::math::convolveLinear(
    OutMaskedImageT& convolvedImage,        ///< convolved image
    InMaskedImageT const& inImage,          ///< image to convolve
    lsst::afw::math::LinearCombinationKernel const& kernel, ///< convolution kernel
    int edgeBit				///< mask bit to indicate pixel includes edge-extended data;
                                        ///< if negative then no bit is set
) {
    if (!kernel.isSpatiallyVarying()) {
        return lsst::afw::math::convolve(convolvedImage, inImage, kernel, edgeBit, false);
    }

    if (convolvedImage.dimensions() != inImage.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if (inImage.dimensions() < kernel.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter(
            "inImage smaller than kernel in columns and/or rows");
    }
    
    typedef lsst::afw::image::MaskedImage<double> BasisImage;
    typedef BasisImage::Ptr BasisImagePtr;
    typedef typename BasisImage::x_iterator BasisX_iterator;
    typedef std::vector<BasisX_iterator> BasisX_iteratorList;
    typedef typename OutMaskedImageT::x_iterator cnvX_iterator;
    typedef lsst::afw::math::LinearCombinationKernel::KernelList kernelListType;

    int const imWidth = inImage.getWidth();
    int const imHeight = inImage.getHeight();
    int const cnvWidth = imWidth + 1 - kernel.getWidth();
    int const cnvHeight = imHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();
    int const cnvEndX = cnvStartX + cnvWidth;  // end index + 1
    int const cnvEndY = cnvStartY + cnvHeight; // end index + 1

    // create a vector of pointers to basis images (input Image convolved with each basis kernel)
    std::vector<BasisImagePtr> basisImagePtrList;
    {
        kernelListType basisKernelList = kernel.getKernelList();
        for (typename kernelListType::const_iterator basisKernelIter = basisKernelList.begin();
             basisKernelIter != basisKernelList.end(); ++basisKernelIter) {
            BasisImagePtr basisImagePtr(new BasisImage(inImage.dimensions()));
            lsst::afw::math::basicConvolve(*basisImagePtr, inImage, **basisKernelIter, false);
            basisImagePtrList.push_back(basisImagePtr);
        }
    }
    BasisX_iteratorList basisIterList;                           // x_iterators for each basis image
    basisIterList.reserve(basisImagePtrList.size());
    
    // iterate over matching pixels of all images to compute output image
    std::vector<double> kernelCoeffList(kernel.getNKernelParameters()); // weights of basic images at this point
    for (int cnvY = cnvStartY; cnvY < cnvEndY; ++cnvY) {
        double const rowPos = lsst::afw::image::indexToPosition(cnvY);
    
        cnvX_iterator cnvXIter = convolvedImage.row_begin(cnvY) + cnvStartX;
        for (unsigned int i = 0; i != basisIterList.size(); ++i) {
            basisIterList[i] = basisImagePtrList[i]->row_begin(cnvY) + cnvStartX;
        }

        for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++cnvXIter) {
            double const colPos = lsst::afw::image::indexToPosition(cnvX);
            
            kernel.computeKernelParametersFromSpatialModel(kernelCoeffList, colPos, rowPos);

            double cnvImagePix = 0.0;
            long cnvMaskPix = 0;
            double cnvVariancePix = 0.0;
            for (unsigned int i = 0; i != basisIterList.size(); ++i) {
                cnvImagePix    += basisIterList[i].image()*kernelCoeffList[i];
                cnvMaskPix     |= basisIterList[i].mask();
                cnvVariancePix += basisIterList[i].variance()*kernelCoeffList[i]*kernelCoeffList[i];

                ++basisIterList[i];
            }
            cnvXIter.image() = cnvImagePix;
            cnvXIter.mask() = cnvMaskPix;
            cnvXIter.variance() = cnvVariancePix;
        }
    }

    _copyBorder(convolvedImage, inImage, kernel, edgeBit);
}

/**
 * @brief Convolve a MaskedImage with a LinearCombinationKernel, returning a new image.
 *
 * @return the convolved MaskedImage.
 *
 * See documentation for the version of convolveLinear that sets pixels in an existing image.
 *
 * @note This function should probably be retired;  it's easily coded by the user in 2 lines
 *
 * @throw lsst::pex::exceptions::InvalidParameter if inImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename ImageT>
ImageT lsst::afw::math::convolveLinearNew(
    ImageT const &inImage,                                     ///< image to convolve
    lsst::afw::math::LinearCombinationKernel const &kernel,    ///< convolution kernel
    int edgeBit                         ///< mask bit to indicate pixel includes edge-extended data;
                                        ///< if negative then no bit is set
) {
    ImageT convolvedImage(inImage.dimensions());
    lsst::afw::math::convolveLinear(convolvedImage, inImage, kernel, edgeBit);
    return convolvedImage;
}
