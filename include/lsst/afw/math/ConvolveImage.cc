// -*- LSST-C++ -*-
/**
 * @file
 *
 * @brief Definition of functions declared in ConvolveImage.h
 *
 * This file is meant to be included by lsst/afw/math/KernelFunctions.h
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

// Private functions to copy Images' borders
#include "lsst/afw/math/copyEdges.h"

/**
 * @brief Apply convolution kernel to an image at one point
 *
 * @note: this is a high performance routine; the user is expected to:
 * - figure out the kernel center and adjust the supplied pixel accessors accordingly
 * For an example of how to do this see the convolve function.
 *
 * @ingroup afw
 */
template <typename OutPixelT, typename InImageT>
inline void lsst::afw::math::apply(
    OutPixelT &outValue,                ///< output image pixel value
    typename InImageT::const_xy_locator const &imageLocator,
                                        ///< locator for image pixel that overlaps (0,0) pixel of kernel(!)
    lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::const_xy_locator const &kernelLocator,
                                        ///< accessor for (0,0) pixel of kernel
    int kwidth,                         ///< number of columns in kernel
    int kheight                         ///< number of rows in kernel
) {
    typedef typename lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::const_xy_locator kernelLocatorType;
    double outImage = 0;
    
    for (int y = 0; y != kheight; ++y) {
        for (int x = 0; x != kwidth; ++x, ++imageLocator.x(), ++kernelLocator.x()) {
            outImage += static_cast<OutPixelT>(imageLocator[0]*kernelLocator[0]);
        }

        imageLocator  += lsst::afw::image::details::difference_type(-kwidth, 1);
        kernelLocator += lsst::afw::image::details::difference_type(-kwidth, 1);
    }

    outValue = outImage;
}

/**
 * @brief Apply separable convolution kernel to an image at one point
 *
 * @note: this is a high performance routine; the user is expected to:
 * - figure out the kernel center and adjust the supplied pixel accessors accordingly
 * For an example of how to do this see the convolve function.
 *
 * @ingroup afw
 */
template <typename OutPixelT, typename InImageT>
inline void lsst::afw::math::apply(
    OutPixelT &outValue,                ///< output pixel value
    typename InImageT::const_xy_locator const &imageLocator,
                                        ///< accessor to for image pixel that overlaps (0,0) pixel of kernel(!)
    std::vector<lsst::afw::math::Kernel::PixelT> const &kernelXList,  ///< kernel column vector
    std::vector<lsst::afw::math::Kernel::PixelT> const &kernelYList   ///< kernel row vector
) {
    typedef typename std::vector<lsst::afw::math::Kernel::PixelT>::const_iterator k_iter;

    std::vector<lsst::afw::math::Kernel::PixelT>::const_iterator kernelYIter = kernelYList.begin();

    double outImage = 0;
    for (k_iter kernelYIter = kernelYList.begin(), end = kernelYList.end();
         kernelYIter != end; ++kernelYIter) {

        double outImageY = 0;
        for (k_iter kernelXIter = kernelXList.begin(), end = kernelXList.end();
             kernelXIter != end; ++kernelXIter, ++imageLocator.x()) {
            double kernelXValue = static_cast<double> (*kernelXIter);
            outImageY += kernelXValue*imageLocator[0];
        }

        outImage += static_cast<double>(*kernelYIter) * outImageY;

        imageLocator += lsst::afw::image::details::difference_type(-kernelXList.size(), 1);
    }
    
    outValue = static_cast<OutPixelT>(outImage);
}

/**
 * @brief Low-level convolution function that does not set edge pixels.
 *
 * convolvedImage must be the same size as inImage.
 * convolvedImage has a border in which the output pixels are not set. This border has size:
 * * kernel.getCtrX/Y() along the left/bottom edge
 * * kernel.getWidth/Height() - 1 - kernel.getCtrX/Y() along the right/top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameter if convolvedImage is not the same size as inImage.
 * @throw lsst::pex::exceptions::InvalidParameter if inImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void lsst::afw::math::basicConvolve(
    OutImageT &convolvedImage,                  ///< convolved image
    InImageT const& inImage,                    ///< image to convolve
    lsst::afw::math::Kernel const& kernel,      ///< convolution kernel
    bool doNormalize                            ///< if True, normalize the kernel, else use "as is"
) {
    typedef typename lsst::afw::math::Kernel::PixelT KernelPixelT;

    typedef typename lsst::afw::image::Image<KernelPixelT>::const_xy_locator kXY_locator;
    typedef typename InImageT::const_xy_locator inXY_locator;
    typedef typename OutImageT::x_iterator cnvX_iterator;

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
                lsst::afw::math::apply<OutImageT, InImageT>(*cnvImIter, inImLoc, kernelLoc, kWidth, kHeight);
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
                lsst::afw::math::apply<OutImageT, InImageT>(*cnvImIter, inImLoc, kernelLoc, kWidth, kHeight);
            }
        }
    }
}
//
// Use the versions in ConvolveMaskedImage.cc
#if 0
/**
 * @brief A version of basicConvolve that should be used when convolving delta function kernels
 */
template <typename OutImageT, typename InImageT>
void lsst::afw::math::basicConvolve(
    OutImageT &convolvedImage,          ///< convolved image
    InImageT const &inImage,            ///< image to convolve
    lsst::afw::math::DeltaFunctionKernel const &kernel,    ///< convolution kernel
    bool doNormalize                                       ///< if True, normalize the kernel, else use "as is"
) {
    assert (!kernel.isSpatiallyVarying());

    typedef typename InImageT::pixel_accessor InLocator;
    typedef typename OutImageT::pixel_accessor OutLocator;

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
    int const cnvWidth = static_cast<int>(inImageWidth) + 1 - static_cast<int>(kernel.getWidth());
    int const cnvHeight = static_cast<int>(inImageHeight) + 1 - static_cast<int>(kernel.getHeight());
    int const cnvStartX = static_cast<int>(kernel.getCtrX());
    int const cnvStartY = static_cast<int>(kernel.getCtrY());
    int const inStartX = kernel.getPixel().first;
    int const inStartY = kernel.getPixel().second;

    // create input and output image accessors
    // and advance each to the right spot
    InLocator inImageYAcc = inImage.origin();
    inImageYAcc.advance(inStartX, inStartY);
    OutLocator cnvImageYAcc = convolvedImage.origin();
    cnvImageYAcc.advance(cnvStartX, cnvStartY);

    lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially invariant delta function basis");
    for (int i = 0; i < cnvHeight; ++i, cnvImageYAcc.next_row(), inImageYAcc.next_row()) {
        InLocator inImageXAcc = inImageYAcc;
        OutLocator cnvImageXAcc = cnvImageYAcc;
        for (int j = 0; j < cnvWidth; ++j, inImageXAcc.next_col(), cnvImageXAcc.next_col()) {
            *cnvImageXAcc = *inImageXAcc;
        }
    }
}

/**
 * @brief A version of basicConvolve that should be used when convolving separable kernels
 */
template <typename OutImageT, typename InImageT>
void lsst::afw::math::basicConvolve(
    OutImageT &convolvedImage,          ///< convolved image
    InImageT const &inImage,            ///< image to convolve
    lsst::afw::math::SeparableKernel const &kernel,  ///< convolution kernel
    bool doNormalize                    ///< if True, normalize the kernel, else use "as is"
) {
    typedef typename lsst::afw::math::Kernel::PixelT KernelPixelT;
    typedef typename InImageT::pixel_accessor InLocator;
    typedef typename OutImageT::pixel_accessor OutLocator;

    if (convolvedImage.dimensions() != inImage.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if (convolvedImage.dimensions() < kernel.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter(
            "inImage smaller than kernel in columns and/or rows");
    }
    
    int const inImageWidth = inImage.getWidth();
    int const inImageHeight = inImage.getHeight();
    int const kWidth = kernel.getWidth();
    int const kHeight = kernel.getHeight();
    int const cnvWidth = static_cast<int>(inImageWidth) + 1 - static_cast<int>(kernel.getWidth());
    int const cnvHeight = static_cast<int>(inImageHeight) + 1 - static_cast<int>(kernel.getHeight());
    int const cnvStartX = static_cast<int>(kernel.getCtrX());
    int const cnvStartY = static_cast<int>(kernel.getCtrY());
    int const cnvEndX = cnvStartX + static_cast<int>(cnvWidth); // end index + 1
    int const cnvEndY = cnvStartY + static_cast<int>(cnvHeight); // end index + 1

    // create input and output image accessors
    // and advance output accessor to lower left pixel that is set by convolution
    InLocator inImageYAcc = inImage.origin();
    OutLocator cnvImageYAcc = convolvedImage.origin();
    cnvImageYAcc.advance(cnvStartX, cnvStartY);
    
    std::vector<lsst::afw::math::Kernel::PixelT> kXVec(kernel.getWidth());
    std::vector<lsst::afw::math::Kernel::PixelT> kYVec(kernel.getHeight());
    
    if (kernel.isSpatiallyVarying()) {
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially varying separable kernel");
        for (int cnvY = cnvStartY; cnvY < cnvEndY;
            ++cnvY, cnvImageYAcc.next_row(), inImageYAcc.next_row()) {
            double rowPos = lsst::afw::image::indexToPosition(cnvY);
            InLocator inImageXAcc = inImageYAcc;
            OutLocator cnvImageXAcc = cnvImageYAcc;
            for (int cnvX = cnvStartX; cnvX < cnvEndX;
                ++cnvX, inImageXAcc.next_col(), cnvImageXAcc.next_col()) {
                double colPos = lsst::afw::image::indexToPosition(cnvX);
                KernelPixelT kSum;
                kernel.computeVectors(kXVec, kYVec, kSum, doNormalize, colPos, rowPos);
                lsst::afw::math::apply(*cnvImageXAcc, inImageXAcc, kXVec, kYVec);
                if (doNormalize) {
                    *cnvImageXAcc /= kSum;
                }
            }
        }
    } else {
        // kernel is spatially invariant
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially invariant separable kernel");
        KernelPixelT kSum;
        kernel.computeVectors(kXVec, kYVec, kSum, doNormalize);
        for (int cnvY = cnvStartY; cnvY < cnvEndY;
            ++cnvY, cnvImageYAcc.next_row(), inImageYAcc.next_row()) {
            InLocator inImageXAcc = inImageYAcc;
            OutLocator cnvImageXAcc = cnvImageYAcc;
            for (int cnvX = cnvStartX; cnvX < cnvEndX;
                ++cnvX, inImageXAcc.next_col(), cnvImageXAcc.next_col()) {
                // g++ 3.6.4 requires the template arguments here to find the function; I don't know why
                lsst::afw::math::apply(*cnvImageXAcc, inImageXAcc, kXVec, kYVec);
            }
        }
    }
}
#endif

/**
 * @brief Convolve an Image with a Kernel, setting pixels of an existing image
 *
 * convolvedImage must be the same size as inImage.
 * convolvedImage has a border in which the output pixels are just a copy of the input pixels.
 * This border has size:
 * * kernel.getCtrX/Y() along the left/bottom edge
 * * kernel.getWidth/Height() - 1 - kernel.getCtrY/Y() along the right/top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameter if convolvedImage is not the same size as inImage.
 * @throw lsst::pex::exceptions::InvalidParameter if inImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT, typename KernelT>
void lsst::afw::math::convolve(
    OutImageT &convolvedImage,          ///< convolved image
    InImageT const &inImage,            ///< image to convolve
    KernelT const &kernel,              ///< convolution kernel
    bool doNormalize                    ///< if True, normalize the kernel, else use "as is"
) {
    lsst::afw::math::basicConvolve(convolvedImage, inImage, kernel, doNormalize);
    _copyBorder(convolvedImage, inImage, kernel);
}

/**
 * @brief Convolve an Image with a Kernel, returning a new image.
 *
 * @return the convolved Image.
 *
 * The returned Image is the same size as inImage.
 * It has a border in which the output pixels are just a copy of the input pixels.
 * This border will have size:
 * * kernel.getCtrX/Y() along the left/bottom edge
 * * kernel.getWidth/Height() - 1 - kernel.getCtrX/Y() along the right/top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameter if inImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename InImageT, typename KernelT>
InImageT lsst::afw::math::convolveNew(
    InImageT const &inImage,    ///< image to convolve
    KernelT const &kernel,  ///< convolution kernel
    bool doNormalize        ///< if True, normalize the kernel, else use "as is"
) {
    InImageT convolvedImage(inImage.getWidth(), inImage.getHeight());
    lsst::afw::math::convolve(convolvedImage, inImage, kernel, doNormalize);
    return convolvedImage;
}

/**
 * @brief Convolve an Image with a LinearCombinationKernel, setting pixels of an existing image.
 *
 * A variant of the convolve function that is faster for spatially varying LinearCombinationKernels.
 * For the sake of speed the kernel is NOT normalized. If you want normalization then call the standard
 * convolve function.
 *
 * The Algorithm:
 * Convolves the input Image by each basis kernel in turn, creating a set of basis images.
 * Then for each output pixel it solves the spatial model and computes the the pixel as
 * the appropriate linear combination of basis images.
 *
 * @throw lsst::pex::exceptions::InvalidParameter if convolvedImage is not the same size as inImage.
 * @throw lsst::pex::exceptions::InvalidParameter if inImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void lsst::afw::math::convolveLinear(
    OutImageT &convolvedImage,          ///< convolved image
    InImageT const &inImage,            ///< image to convolve
    lsst::afw::math::LinearCombinationKernel const& kernel ///< convolution kernel
) {
    if (!kernel.isSpatiallyVarying()) {
        return lsst::afw::math::convolve(convolvedImage, inImage, kernel, false);
    }

    int const imWidth = inImage.getWidth();
    int const imHeight = inImage.getHeight();
    int const kWidth = kernel.getWidth();
    int const kHeight = kernel.getHeight();
    if (convolvedImage.dimensions() != inImage.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if (inImage.dimensions() < kernel.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("inImage smaller than kernel in columns and/or rows");
    }
    
    typedef lsst::afw::image::Image<double> BasisImage;
    typedef BasisImage::Ptr BasisImagePtr;
    typedef typename BasisImage::x_iterator BasisX_iterator;
    typedef std::vector<BasisX_iterator> BasisX_iteratorList;
    typedef typename OutImageT::x_iterator cnvX_iterator;
    typedef lsst::afw::math::LinearCombinationKernel::KernelList kernelListType;

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
    BasisX_iteratorList basisIterList(basisImagePtrList.size()); // x_iterators for each basis image
    
    // iterate over matching pixels of all images to compute output image
    std::vector<double> kernelCoeffList(kernel.getNKernelParameters()); // weights of basic images at this point
    for (int cnvY = cnvStartY; cnvY < cnvEndY; ++cnvY) {
        double const rowPos = lsst::afw::image::indexToPosition(cnvY);
    
        cnvX_iterator cnvXIter = convolvedImage.row_begin(cnvY) + cnvStartX;
        for (int i = 0; i != basisIterList.size(); ++i) {
            basisIterList[i] = basisImagePtrList[i]->row_begin(cnvY) + cnvStartX;
        }

        for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++cnvXIter) {
            double const colPos = lsst::afw::image::indexToPosition(cnvX);
            
            kernel.computeKernelParametersFromSpatialModel(kernelCoeffList, colPos, rowPos);

            double cnvImagePix = 0.0;
            for (int i = 0; i != basisIterList.size(); ++i) {
                cnvImagePix += kernelCoeffList[i]*(*basisIterList[i]);
                ++basisIterList[i];
            }
            *cnvXIter = cnvImagePix;
        }
    }

    _copyBorder(convolvedImage, inImage, kernel);
}

/**
 * @brief Convolve an Image with a LinearCombinationKernel, returning a new image.
 *
 * @return the convolved Image.
 *
 * See documentation for the version of convolveLinear that sets pixels in an existing image.
 *
 * @throw lsst::pex::exceptions::InvalidParameter if inImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename ImageT>
ImageT lsst::afw::math::convolveLinearNew(
    ImageT const &inImage,                                  ///< image to convolve
    lsst::afw::math::LinearCombinationKernel const &kernel  ///< convolution kernel
) {
    ImageT convolvedImage(inImage.getWidth(), inImage.getHeight());
    lsst::afw::math::convolveLinear(convolvedImage, inImage, kernel);
    return convolvedImage;
}
