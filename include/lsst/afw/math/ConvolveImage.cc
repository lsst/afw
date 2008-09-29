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
template <typename OutPixelT, typename InPixelT>
inline void lsst::afw::math::apply(
    OutPixelT &outValue,    ///< output image pixel value
    typename lsst::afw::image::Image<InPixelT>::const_xy_locator const &imageAccessor,
    ///< accessor to for image pixel that overlaps (0,0) pixel of kernel(!)
    lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::const_xy_locator const &kernelAccessor,
                                        ///< accessor for (0,0) pixel of kernel
    unsigned int kwidth,                 ///< number of columns in kernel
    unsigned int kheight                 ///< number of rows in kernel
) {
    typedef typename lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::const_xy_locator kernelAccessorType;
    double outImage = 0;
    typename lsst::afw::image::Image<InPixelT>::xy_locator imageRowAcc = imageAccessor;
    
    for (int y = 0; y != kheight; ++y) {
        for (int x = 0; x != kwidth; ++x, ++imageAccessor.x(), ++kernelAccessor.x()) {
            outImage += static_cast<OutPixelT>(imageAccessor[0]*kernelAccessor[0]);
        }

        imageAccessor  += lsst::afw::image::details::difference_type(-kwidth, 1);
        kernelAccessor += lsst::afw::image::details::difference_type(-kwidth, 1);
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
template <typename OutPixelT, typename InPixelT>
inline void lsst::afw::math::apply(
    OutPixelT &outValue,    ///< output pixel value
    typename lsst::afw::image::Image<InPixelT>::const_xy_locator const &imageAccessor,
        ///< accessor to for image pixel that overlaps (0,0) pixel of kernel(!)
    std::vector<lsst::afw::math::Kernel::PixelT> const &kernelColList,  ///< kernel column vector
    std::vector<lsst::afw::math::Kernel::PixelT> const &kernelRowList   ///< kernel row vector
) {
    typedef typename std::vector<lsst::afw::math::Kernel::PixelT>::const_iterator k_iter;

    std::vector<lsst::afw::math::Kernel::PixelT>::const_iterator kernelRowIter = kernelRowList.begin();

    double outImage = 0;
    for (k_iter kernelRowIter = kernelRowList.begin(), end = kernelRowList.end();
         kernelRowIter != end; ++kernelRowIter) {

        double outImageRow = 0;
        for (k_iter kernelColIter = kernelColList.begin(), end = kernelColList.end();
             kernelColIter != end; ++kernelColIter, ++imageAccessor.x()) {
            double kernelColValue = static_cast<double> (*kernelColIter);
            outImageRow += kernelColValue*imageAccessor[0];
        }

        outImage += static_cast<double>(*kernelRowIter) * outImageRow;

        imageAccessor += lsst::afw::image::details::difference_type(-kernelColList.size(), 1);
    }
    
    outValue = static_cast<OutPixelT>(outImage);
}

/**
 * @brief Low-level convolution function that does not set edge pixels.
 *
 * convolvedImage must be the same size as inImage.
 * convolvedImage has a border in which the output pixels are not set. This border has size:
 * * kernel.getCtrCol/Row() along the left/bottom edge
 * * kernel.getWidth/Height() - 1 - kernel.getCtrCol/Row() along the right/top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameter if convolvedImage is not the same size as inImage.
 * @throw lsst::pex::exceptions::InvalidParameter if inImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename OutPixelT, typename InPixelT>
void lsst::afw::math::basicConvolve(
    lsst::afw::image::Image<OutPixelT> &convolvedImage,       ///< convolved image
    lsst::afw::image::Image<InPixelT> const &inImage,    ///< image to convolve
    lsst::afw::math::Kernel const &kernel,  ///< convolution kernel
    bool doNormalize                        ///< if True, normalize the kernel, else use "as is"
) {
    typedef typename lsst::afw::math::Kernel::PixelT KernelPixelT;

    typedef typename lsst::afw::image::Image<KernelPixelT>::const_xy_locator kXY_locator;
    typedef typename lsst::afw::image::Image<InPixelT>::const_xy_locator inXY_locator;
    typedef typename lsst::afw::image::Image<OutPixelT>::x_iterator cnvX_iterator;

    int const inImageWidth = inImage.getWidth();
    int const inImageHeight = inImage.getHeight();
    int const kWidth = kernel.getWidth();
    int const kHeight = kernel.getHeight();
    if (convolvedImage.dimensions() != inImage.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if (inImage.dimensions() < kernel.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("inImage smaller than kernel in columns and/or rows");
    }
    
    int const cnvWidth = inImageWidth + 1 - kernel.getWidth();
    int const cnvHeight = inImageHeight + 1 - kernel.getHeight();
    int const cnvStartCol = kernel.getCtrX();
    int const cnvStartRow = kernel.getCtrY();
    int const cnvEndCol = cnvStartCol + cnvWidth;  // end index + 1
    int const cnvEndRow = cnvStartRow + cnvHeight; // end index + 1

    // create input and output image accessors
    // and advance output accessor to lower left pixel that is set by convolution
    if (kernel.isSpatiallyVarying()) {
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is spatially varying");

        lsst::afw::image::Image<KernelPixelT> kernelImage(kernel.dimensions()); // the kernel at a point

        for (int cnvRow = cnvStartRow; cnvRow != cnvEndRow; ++cnvRow) {
            double const rowPos = lsst::afw::image::indexToPosition(cnvRow);
            
            inXY_locator  inImLoc =  inImage.at_xy(0, cnvRow - cnvStartRow);
            cnvX_iterator cnvImIter = convolvedImage.row_begin(cnvRow) + cnvStartCol;
            for (int cnvCol = cnvStartCol; cnvCol != cnvEndCol; ++cnvCol, ++inImLoc.x(), ++cnvImIter) {
                KernelPixelT kSum;
                double const colPos = lsst::afw::image::indexToPosition(cnvCol);
                kernel.computeImage(kernelImage, kSum, false, colPos, rowPos);
                // g++ 3.6.4 requires the template arguments here to find the function; I don't know why
                // Is this still true? RHL
                kXY_locator kernelLoc = kernelImage.xy_at(0,0);
                lsst::afw::math::apply<OutPixelT, InPixelT>(*cnvImIter, inImLoc, kernelLoc, kWidth, kHeight);
                if (doNormalize) {
                    *cnvImIter /= kSum;
                }
            }
        }
    } else {
        // kernel is spatially invariant
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is spatially invariant");
        KernelPixelT kSum;
        lsst::afw::image::Image<KernelPixelT> kernelImage = kernel.computeNewImage(kSum, doNormalize);

        for (int cnvRow = cnvStartRow; cnvRow != cnvEndRow; ++cnvRow) {
            inXY_locator inImLoc =  inImage.at_xy(0, cnvRow - cnvStartRow);
            for (cnvX_iterator cnvImIter = convolvedImage.row_begin(cnvRow) + cnvStartCol,
                     cnvImEnd = cnvImIter + cnvEndCol - cnvStartCol; cnvImIter != cnvImEnd; ++inImLoc.x(), ++cnvImIter) {
                kXY_locator kernelLoc = kernelImage.xy_at(0,0);
                lsst::afw::math::apply<OutPixelT, InPixelT>(*cnvImIter, inImLoc, kernelLoc, kWidth, kHeight);
            }
        }
    }
}

/**
 * @brief A version of basicConvolve that should be used when convolving delta function kernels
 */
template <typename OutPixelT, typename InPixelT>
void lsst::afw::math::basicConvolve(
    lsst::afw::image::Image<OutPixelT> &convolvedImage,       ///< convolved image
    lsst::afw::image::Image<InPixelT> const &inImage,    ///< image to convolve
    lsst::afw::math::DeltaFunctionKernel const &kernel,    ///< convolution kernel
    bool doNormalize    ///< if True, normalize the kernel, else use "as is"
) {
    assert (!kernel.isSpatiallyVarying());

    typedef typename lsst::afw::image::Image<InPixelT>::pixel_accessor InPixelAccessor;
    typedef typename lsst::afw::image::Image<OutPixelT>::pixel_accessor OutPixelAccessor;

    int const inImageWidth = inImage.getWidth();
    int const inImageHeight = inImage.getHeight();
    int const kWidth = kernel.getWidth();
    int const kHeight = kernel.getHeight();

    if (convolvedImage.getWidth() != inImageWidth || convolvedImage.getHeight() != inImageHeight) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if ((inImageWidth < kWidth) || (inImageHeight < kHeight)) {
        throw lsst::pex::exceptions::InvalidParameter("inImage smaller than kernel in columns and/or rows");
    }
    
    int const cnvWidth = static_cast<int>(inImageWidth) + 1 - static_cast<int>(kernel.getWidth());
    int const cnvHeight = static_cast<int>(inImageHeight) + 1 - static_cast<int>(kernel.getHeight());
    int const cnvStartCol = static_cast<int>(kernel.getCtrX());
    int const cnvStartRow = static_cast<int>(kernel.getCtrY());
    int const inStartCol = kernel.getPixel().first;
    int const inStartRow = kernel.getPixel().second;

    // create input and output image accessors
    // and advance each to the right spot
    InPixelAccessor inImageRowAcc = inImage.origin();
    inImageRowAcc.advance(inStartCol, inStartRow);
    OutPixelAccessor cnvImageRowAcc = convolvedImage.origin();
    cnvImageRowAcc.advance(cnvStartCol, cnvStartRow);

    lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially invariant delta function basis");
    for (int i = 0; i < cnvHeight; ++i, cnvImageRowAcc.next_row(), inImageRowAcc.next_row()) {
        InPixelAccessor inImageColAcc = inImageRowAcc;
        OutPixelAccessor cnvImageColAcc = cnvImageRowAcc;
        for (int j = 0; j < cnvWidth; ++j, inImageColAcc.next_col(), cnvImageColAcc.next_col()) {
            *cnvImageColAcc = *inImageColAcc;
        }
    }
}

/**
 * @brief A version of basicConvolve that should be used when convolving separable kernels
 */
template <typename OutPixelT, typename InPixelT>
void lsst::afw::math::basicConvolve(
    lsst::afw::image::Image<OutPixelT> &convolvedImage,       ///< convolved image
    lsst::afw::image::Image<InPixelT> const &inImage,    ///< image to convolve
    lsst::afw::math::SeparableKernel const &kernel,  ///< convolution kernel
    bool doNormalize                    ///< if True, normalize the kernel, else use "as is"
) {
    typedef typename lsst::afw::math::Kernel::PixelT KernelPixelT;
    typedef typename lsst::afw::image::Image<InPixelT>::pixel_accessor InPixelAccessor;
    typedef typename lsst::afw::image::Image<OutPixelT>::pixel_accessor OutPixelAccessor;

    int const inImageWidth = inImage.getWidth();
    int const inImageHeight = inImage.getHeight();
    int const kWidth = kernel.getWidth();
    int const kHeight = kernel.getHeight();
    if ((convolvedImage.getWidth() != inImageWidth) || (convolvedImage.getHeight() != inImageHeight)) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if ((inImageWidth < kWidth) || (inImageHeight < kHeight)) {
        throw lsst::pex::exceptions::InvalidParameter("inImage smaller than kernel in columns and/or rows");
    }
    
    int const cnvWidth = static_cast<int>(inImageWidth) + 1 - static_cast<int>(kernel.getWidth());
    int const cnvHeight = static_cast<int>(inImageHeight) + 1 - static_cast<int>(kernel.getHeight());
    int const cnvStartCol = static_cast<int>(kernel.getCtrX());
    int const cnvStartRow = static_cast<int>(kernel.getCtrY());
    int const cnvEndCol = cnvStartCol + static_cast<int>(cnvWidth); // end index + 1
    int const cnvEndRow = cnvStartRow + static_cast<int>(cnvHeight); // end index + 1

    // create input and output image accessors
    // and advance output accessor to lower left pixel that is set by convolution
    InPixelAccessor inImageRowAcc = inImage.origin();
    OutPixelAccessor cnvImageRowAcc = convolvedImage.origin();
    cnvImageRowAcc.advance(cnvStartCol, cnvStartRow);
    
    std::vector<lsst::afw::math::Kernel::PixelT> kColVec(kernel.getWidth());
    std::vector<lsst::afw::math::Kernel::PixelT> kRowVec(kernel.getHeight());
    
    if (kernel.isSpatiallyVarying()) {
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially varying separable kernel");
        for (int cnvRow = cnvStartRow; cnvRow < cnvEndRow;
            ++cnvRow, cnvImageRowAcc.next_row(), inImageRowAcc.next_row()) {
            double rowPos = lsst::afw::image::indexToPosition(cnvRow);
            InPixelAccessor inImageColAcc = inImageRowAcc;
            OutPixelAccessor cnvImageColAcc = cnvImageRowAcc;
            for (int cnvCol = cnvStartCol; cnvCol < cnvEndCol;
                ++cnvCol, inImageColAcc.next_col(), cnvImageColAcc.next_col()) {
                double colPos = lsst::afw::image::indexToPosition(cnvCol);
                KernelPixelT kSum;
                kernel.computeVectors(kColVec, kRowVec, kSum, doNormalize, colPos, rowPos);
                // g++ 3.6.4 requires the template arguments here to find the function; I don't know why
                // Is this still true? RHL
                lsst::afw::math::apply<OutPixelT, InPixelT>(*cnvImageColAcc, inImageColAcc, kColVec, kRowVec);
                if (doNormalize) {
                    *(cnvImageColAcc) /= static_cast<InPixelT>(kSum);
                }
            }
        }
    } else {
        // kernel is spatially invariant
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially invariant separable kernel");
        KernelPixelT kSum;
        kernel.computeVectors(kColVec, kRowVec, kSum, doNormalize);
        for (int cnvRow = cnvStartRow; cnvRow < cnvEndRow;
            ++cnvRow, cnvImageRowAcc.next_row(), inImageRowAcc.next_row()) {
            InPixelAccessor inImageColAcc = inImageRowAcc;
            OutPixelAccessor cnvImageColAcc = cnvImageRowAcc;
            for (int cnvCol = cnvStartCol; cnvCol < cnvEndCol;
                ++cnvCol, inImageColAcc.next_col(), cnvImageColAcc.next_col()) {
                // g++ 3.6.4 requires the template arguments here to find the function; I don't know why
                lsst::afw::math::apply<OutPixelT, InPixelT>(*cnvImageColAcc, inImageColAcc, kColVec, kRowVec);
            }
        }
    }
}

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
template <typename OutPixelT, typename InPixelT, typename KernelT>
void lsst::afw::math::convolve(
    lsst::afw::image::Image<OutPixelT> &convolvedImage,       ///< convolved image
    lsst::afw::image::Image<InPixelT> const &inImage,    ///< image to convolve
    KernelT const &kernel,  ///< convolution kernel
    bool doNormalize        ///< if True, normalize the kernel, else use "as is"
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
template <typename InPixelT, typename KernelT>
lsst::afw::image::Image<InPixelT> lsst::afw::math::convolveNew(
    lsst::afw::image::Image<InPixelT> const &inImage,    ///< image to convolve
    KernelT const &kernel,  ///< convolution kernel
    bool doNormalize        ///< if True, normalize the kernel, else use "as is"
) {
    lsst::afw::image::Image<InPixelT> convolvedImage(inImage.getWidth(), inImage.getHeight());
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
template <typename OutPixelT, typename InPixelT>
void lsst::afw::math::convolveLinear(
    lsst::afw::image::Image<OutPixelT> &convolvedImage,     ///< convolved image
    lsst::afw::image::Image<InPixelT> const &inImage,       ///< image to convolve
    lsst::afw::math::LinearCombinationKernel const &kernel  ///< convolution kernel
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
    typedef typename lsst::afw::image::Image<double>::x_iterator BasisX_iterator;
    typedef std::vector<BasisX_iterator> BasisX_iteratorList;
    typedef typename lsst::afw::image::Image<OutPixelT>::x_iterator cnvX_iterator;
    typedef lsst::afw::math::LinearCombinationKernel::KernelList kernelListType;

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

    int const cnvWidth = imWidth + 1 - kernel.getWidth();
    int const cnvHeight = imHeight + 1 - kernel.getHeight();
    int const cnvStartCol = kernel.getCtrX();
    int const cnvStartRow = kernel.getCtrY();
    int const cnvEndCol = cnvStartCol + cnvWidth;  // end index + 1
    int const cnvEndRow = cnvStartRow + cnvHeight; // end index + 1
    
    // iterate over matching pixels of all images to compute output image
    std::vector<double> kernelCoeffList(kernel.getNKernelParameters()); // weights of basic images at this point
    for (int cnvRow = cnvStartRow; cnvRow < cnvEndRow; ++cnvRow) {
        double const rowPos = lsst::afw::image::indexToPosition(cnvRow);
    
        cnvX_iterator cnvColIter = convolvedImage.row_begin(cnvRow) + cnvStartCol;
        for (int i = 0; i != basisIterList.size(); ++i) {
            basisIterList[i] = basisImagePtrList[i]->row_begin(cnvRow) + cnvStartCol;
        }

        for (int cnvCol = cnvStartCol; cnvCol != cnvEndCol; ++cnvCol, ++cnvColIter) {
            double const colPos = lsst::afw::image::indexToPosition(cnvCol);
            
            kernel.computeKernelParametersFromSpatialModel(kernelCoeffList, colPos, rowPos);

            double cnvImagePix = 0.0;
            for (int i = 0; i != basisIterList.size(); ++i) {
                cnvImagePix += kernelCoeffList[i]*(*basisIterList[i]);
                ++basisIterList[i];
            }
            *cnvColIter = cnvImagePix;
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
template <typename InPixelT>
lsst::afw::image::Image<InPixelT> lsst::afw::math::convolveLinearNew(
    lsst::afw::image::Image<InPixelT> const &inImage,       ///< image to convolve
    lsst::afw::math::LinearCombinationKernel const &kernel  ///< convolution kernel
) {
    lsst::afw::image::Image<InPixelT> convolvedImage(inImage.getWidth(), inImage.getHeight());
    lsst::afw::math::convolveLinear(convolvedImage, inImage, kernel);
    return convolvedImage;
}
