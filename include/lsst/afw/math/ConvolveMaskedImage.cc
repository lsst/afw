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
template <typename OutPixelT, typename InPixelT, typename MaskPixelT>
inline void lsst::afw::math::apply(
    lsst::afw::image::MaskedPixelAccessor<OutPixelT, MaskPixelT> &outAccessor,    ///< accessor for output pixel
    lsst::afw::image::MaskedPixelAccessor<InPixelT, MaskPixelT> const &maskedImageAccessor,
        ///< accessor to for masked image pixel that overlaps (0,0) pixel of kernel(!)
    lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::pixel_accessor const &kernelAccessor,
        ///< accessor for (0,0) pixel of kernel
    unsigned int cols,  ///< number of columns in kernel
    unsigned int rows   ///< number of rows in kernel
) {
    typedef typename lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::pixel_accessor KernelAccessor;
    double outImage = 0;
    double outVariance = 0;
    MaskPixelT outMask = 0;
    lsst::afw::image::MaskedPixelAccessor<InPixelT, MaskPixelT> mImageRowAcc = maskedImageAccessor;
    KernelAccessor kRow = kernelAccessor;
    for (unsigned int row = 0; row < rows; ++row, mImageRowAcc.nextRow(), kRow.next_row()) {
        InPixelT *imagePtr = &(*(mImageRowAcc.image));
        InPixelT *varPtr = &(*(mImageRowAcc.variance));
        MaskPixelT *maskPtr = &(*(mImageRowAcc.mask));
        // assume data contiguous along rows; use a pointer instead of a vw pixel accessor to gain speed
        lsst::afw::math::Kernel::PixelT *kerPtr = &(*kRow);
        for (unsigned int col = 0; col < cols; ++col, imagePtr++, varPtr++, maskPtr++, kerPtr++) {
            lsst::afw::math::Kernel::PixelT ker = *kerPtr;

#ifdef IgnoreKernelZeroPixels
            if (ker != 0) {
#else
            {
#endif
                outImage += static_cast<double>(ker * (*imagePtr));
                outVariance += static_cast<double>(ker * ker * (*varPtr));
                outMask |= *maskPtr;
            }
        }
//        // this version does not assume data order
//        lsst::afw::image::MaskedPixelAccessor<InPixelT, MaskPixelT> mImageColAcc = mImageRowAcc;
//        KernelAccessor kCol = kRow;
//        for (unsigned int col = 0; col < cols; ++col, mImageColAcc.nextCol(), kCol.next_col()) {
//            Kernel::PixelT ker = *kCol;
//#ifdef IgnoreKernelZeroPixels
//            if (ker != 0) {
//#else
//            {
//#endif
//                outImage += static_cast<double>(ker * (*(mImageColAcc.image)));
//                outVariance += static_cast<double>(ker * ker * (*(mImageColAcc.variance)));
//                outMask |= *(mImageColAcc.mask);
//            }
//        }
    }
    *(outAccessor.image) = static_cast<OutPixelT>(outImage);
    *(outAccessor.variance) = static_cast<OutPixelT>(outVariance);
    *(outAccessor.mask) = outMask;
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
template <typename OutPixelT, typename InPixelT, typename MaskPixelT>
inline void lsst::afw::math::apply(
    lsst::afw::image::MaskedPixelAccessor<OutPixelT, MaskPixelT> &outAccessor,    ///< accessor for output pixel
    lsst::afw::image::MaskedPixelAccessor<InPixelT, MaskPixelT> const &maskedImageAccessor,
        ///< accessor to for masked image pixel that overlaps (0,0) pixel of kernel(!)
    std::vector<lsst::afw::math::Kernel::PixelT> const &kernelColList,  ///< kernel column vector
    std::vector<lsst::afw::math::Kernel::PixelT> const &kernelRowList   ///< kernel row vector
) {
    double outImage = 0;
    double outVariance = 0;
    MaskPixelT outMask = 0;
    lsst::afw::image::MaskedPixelAccessor<InPixelT, MaskPixelT> mImageRowAcc = maskedImageAccessor;
    std::vector<lsst::afw::math::Kernel::PixelT>::const_iterator kernelRowIter = kernelRowList.begin();
    for ( ; kernelRowIter != kernelRowList.end(); ++kernelRowIter, mImageRowAcc.nextRow()) {
        lsst::afw::image::MaskedPixelAccessor<InPixelT, MaskPixelT> mImageColAcc = mImageRowAcc;
        std::vector<lsst::afw::math::Kernel::PixelT>::const_iterator kernelColIter = kernelColList.begin();
        double outImageRow = 0;
        double outVarianceRow = 0;
        MaskPixelT outMaskRow = 0;
        for ( ; kernelColIter != kernelColList.end(); ++kernelColIter, mImageColAcc.nextCol()) {
            double kernelColValue = static_cast<double> (*kernelColIter);
#ifdef IgnoreKernelZeroPixels
            if (kernelColValue != 0) {
#else
            {
#endif
                outImageRow += kernelColValue * (*(mImageColAcc.image));
                outVarianceRow += kernelColValue * kernelColValue * (*(mImageColAcc.variance));
                outMaskRow |= *(mImageColAcc.mask);
            }
        }
        double kernelRowValue = static_cast<double> (*kernelRowIter);
#ifdef IgnoreKernelZeroPixels
        if (kernelRowValue != 0) {
#else
        {
#endif
            outImage += kernelRowValue * outImageRow;
            outVariance += kernelRowValue * kernelRowValue * outVarianceRow;
            outMask |= outMaskRow;
        }
    }
    *(outAccessor.image) = static_cast<OutPixelT>(outImage);
    *(outAccessor.variance) = static_cast<OutPixelT>(outVariance);
    *(outAccessor.mask) = outMask;
}


/**
 * @brief Low-level convolution function that does not set edge pixels.
 *
 * convolvedImage must be the same size as maskedImage.
 * convolvedImage has a border in which the output pixels are not set. This border has size:
 * * kernel.getCtrCol/Row() along the left/bottom edge
 * * kernel.getCols/Rows() - 1 - kernel.getCtrCol/Row() along the right/top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameter if convolvedImage is not the same size as maskedImage.
 * @throw lsst::pex::exceptions::InvalidParameter if maskedImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename OutPixelT, typename InPixelT, typename MaskPixelT>
void lsst::afw::math::basicConvolve(
    lsst::afw::image::MaskedImage<OutPixelT, MaskPixelT> &convolvedImage,       ///< convolved image
    lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> const &maskedImage,    ///< image to convolve
    lsst::afw::math::Kernel const &kernel,  ///< convolution kernel
    bool doNormalize                    ///< if True, normalize the kernel, else use "as is"
) {
    typedef typename lsst::afw::math::Kernel::PixelT KernelPixelT;
    typedef lsst::afw::image::MaskedPixelAccessor<InPixelT, MaskPixelT> InPixelAccessor;
    typedef lsst::afw::image::MaskedPixelAccessor<OutPixelT, MaskPixelT> OutPixelAccessor;
    typedef typename lsst::afw::image::Image<KernelPixelT>::pixel_accessor KernelAccessor;

    const unsigned int mImageColAccs = maskedImage.getCols();
    const unsigned int mImageRowAccs = maskedImage.getRows();
    const unsigned int kCols = kernel.getCols();
    const unsigned int kRows = kernel.getRows();
    if ((convolvedImage.getCols() != mImageColAccs) || (convolvedImage.getRows() != mImageRowAccs)) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as maskedImage");
    }
    if ((mImageColAccs< kCols) || (mImageRowAccs < kRows)) {
        throw lsst::pex::exceptions::InvalidParameter(
            "maskedImage smaller than kernel in columns and/or rows");
    }
    
    const int cnvCols = static_cast<int>(mImageColAccs) + 1 - static_cast<int>(kernel.getCols());
    const int cnvRows = static_cast<int>(mImageRowAccs) + 1 - static_cast<int>(kernel.getRows());
    const int cnvStartCol = static_cast<int>(kernel.getCtrCol());
    const int cnvStartRow = static_cast<int>(kernel.getCtrRow());
    const int cnvEndCol = cnvStartCol + static_cast<int>(cnvCols); // end index + 1
    const int cnvEndRow = cnvStartRow + static_cast<int>(cnvRows); // end index + 1

    // create input and output image accessors
    // and advance output accessor to lower left pixel that is set by convolution
    InPixelAccessor mImageRowAcc(maskedImage);
    OutPixelAccessor cnvRowAcc(convolvedImage);
    cnvRowAcc.advance(cnvStartCol, cnvStartRow);
    
    if (kernel.isSpatiallyVarying()) {
        lsst::afw::image::Image<KernelPixelT> kernelImage(kernel.getCols(), kernel.getRows());
        KernelAccessor kernelAccessor = kernelImage.origin();
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is spatially varying");
        for (int cnvRow = cnvStartRow; cnvRow < cnvEndRow; ++cnvRow, cnvRowAcc.nextRow(), mImageRowAcc.nextRow()) {
            double rowPos = lsst::afw::image::indexToPosition(cnvRow);
            InPixelAccessor mImageColAcc = mImageRowAcc;
            OutPixelAccessor cnvColAcc = cnvRowAcc;
            for (int cnvCol = cnvStartCol; cnvCol < cnvEndCol; ++cnvCol, mImageColAcc.nextCol(), cnvColAcc.nextCol()) {
                double colPos = lsst::afw::image::indexToPosition(cnvCol);
                KernelPixelT kSum;
                kernel.computeImage(kernelImage, kSum, false, colPos, rowPos);
                // g++ 3.6.4 requires the template arguments here to find the function; I don't know why
                // Is this still true? RHL
                lsst::afw::math::apply<OutPixelT, InPixelT, MaskPixelT>(
                    cnvColAcc, mImageColAcc, kernelAccessor, kCols, kRows);
                if (doNormalize) {
                    *(cnvColAcc.image) /= static_cast<OutPixelT>(kSum);
                    *(cnvColAcc.variance) /= static_cast<OutPixelT>(kSum * kSum);
                }
            }
        }
    } else {
        // kernel is spatially invariant
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is spatially invariant");
        KernelPixelT kSum;
        lsst::afw::image::Image<KernelPixelT> kernelImage = kernel.computeNewImage(kSum, doNormalize, 0.0, 0.0);
        KernelAccessor kernelAccessor = kernelImage.origin();
        for (int cnvRow = cnvStartRow; cnvRow < cnvEndRow; ++cnvRow, cnvRowAcc.nextRow(), mImageRowAcc.nextRow()) {
            InPixelAccessor mImageColAcc = mImageRowAcc;
            OutPixelAccessor cnvColAcc = cnvRowAcc;
            for (int cnvCol = cnvStartCol; cnvCol < cnvEndCol; ++cnvCol, mImageColAcc.nextCol(), cnvColAcc.nextCol()) {
                // g++ 3.6.4 requires the template arguments here to find the function; I don't know why
                lsst::afw::math::apply<OutPixelT, InPixelT, MaskPixelT>(
                    cnvColAcc, mImageColAcc, kernelAccessor, kCols, kRows);
            }
        }
    }
}

/************************************************************************************************************/
/**
 * @brief A version of basicConvolve that should be used when convolving delta function kernels
 */
template <typename OutPixelT, typename InPixelT, typename MaskPixelT>
void lsst::afw::math::basicConvolve(
    lsst::afw::image::MaskedImage<OutPixelT, MaskPixelT> &convolvedImage,       ///< convolved image
    lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> const &maskedImage,    ///< image to convolve
    lsst::afw::math::DeltaFunctionKernel const &kernel,    ///< convolution kernel
    bool doNormalize    ///< if True, normalize the kernel, else use "as is"
) {
    assert (!kernel.isSpatiallyVarying());

    typedef typename lsst::afw::image::MaskedPixelAccessor<InPixelT, MaskPixelT> InMaskedImageAccessor;
    typedef typename lsst::afw::image::MaskedPixelAccessor<OutPixelT, MaskPixelT> OutMaskedImageAccessor;
    //
    // It's a pain to deal with unsigned ints when subtracting; so don't
    //
    const int mImageCols = maskedImage.getCols(); // size of input region
    const int mImageRows = maskedImage.getRows();
    const int kCols = kernel.getCols(); // size of Kernel
    const int kRows = kernel.getRows();

    if (static_cast<int>(convolvedImage.getCols()) != mImageCols ||
        static_cast<int>(convolvedImage.getRows()) != mImageRows) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as maskedImage");
    }
    if ((mImageCols< kCols) || (mImageRows < kRows)) {
        throw lsst::pex::exceptions::InvalidParameter(
            "maskedImage smaller than kernel in columns and/or rows");
    }
    
    const int cnvCols = static_cast<int>(mImageCols) + 1 - static_cast<int>(kernel.getCols());
    const int cnvRows = static_cast<int>(mImageRows) + 1 - static_cast<int>(kernel.getRows());
    const int cnvStartCol = static_cast<int>(kernel.getCtrCol());
    const int cnvStartRow = static_cast<int>(kernel.getCtrRow());
    const int inStartCol = kernel.getPixel().first;
    const int inStartRow = kernel.getPixel().second;

    // create input and output image accessors
    // and advance output accessor to lower left pixel that is set by convolution
    InMaskedImageAccessor mImageRowAcc(maskedImage);
    mImageRowAcc.advance(inStartCol, inStartRow);
    OutMaskedImageAccessor cnvRowAcc(convolvedImage);
    cnvRowAcc.advance(cnvStartCol, cnvStartRow);

    lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially invariant delta function basis");
    for (int i = 0; i < cnvRows; ++i, cnvRowAcc.nextRow(), mImageRowAcc.nextRow()) {
        InMaskedImageAccessor mImageColAcc = mImageRowAcc;
        OutMaskedImageAccessor cnvColAcc = cnvRowAcc;
        for (int j = 0; j < cnvCols; ++j, mImageColAcc.nextCol(), cnvColAcc.nextCol()) {
            *cnvColAcc.image =    *mImageColAcc.image;
            *cnvColAcc.variance = *mImageColAcc.variance;
            *cnvColAcc.mask =     *mImageColAcc.mask;
        }
    }
}

/**
 * @brief A version of basicConvolve that should be used when convolving separable kernels
 */
template <typename OutPixelT, typename InPixelT, typename MaskPixelT>
void lsst::afw::math::basicConvolve(
    lsst::afw::image::MaskedImage<OutPixelT, MaskPixelT> &convolvedImage,       ///< convolved image
    lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> const &maskedImage,    ///< image to convolve
    lsst::afw::math::SeparableKernel const &kernel,  ///< convolution kernel
    bool doNormalize                    ///< if True, normalize the kernel, else use "as is"
) {
    typedef typename lsst::afw::math::Kernel::PixelT KernelPixelT;
    typedef lsst::afw::image::MaskedPixelAccessor<InPixelT, MaskPixelT> InPixelAccessor;
    typedef lsst::afw::image::MaskedPixelAccessor<OutPixelT, MaskPixelT> OutPixelAccessor;

    const unsigned int mImageColAccs = maskedImage.getCols();
    const unsigned int mImageRowAccs = maskedImage.getRows();
    const unsigned int kCols = kernel.getCols();
    const unsigned int kRows = kernel.getRows();
    if ((convolvedImage.getCols() != mImageColAccs) || (convolvedImage.getRows() != mImageRowAccs)) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as maskedImage");
    }
    if ((mImageColAccs< kCols) || (mImageRowAccs < kRows)) {
        throw lsst::pex::exceptions::InvalidParameter(
            "maskedImage smaller than kernel in columns and/or rows");
    }
    
    const int cnvCols = static_cast<int>(mImageColAccs) + 1 - static_cast<int>(kernel.getCols());
    const int cnvRows = static_cast<int>(mImageRowAccs) + 1 - static_cast<int>(kernel.getRows());
    const int cnvStartCol = static_cast<int>(kernel.getCtrCol());
    const int cnvStartRow = static_cast<int>(kernel.getCtrRow());
    const int cnvEndCol = cnvStartCol + static_cast<int>(cnvCols); // end index + 1
    const int cnvEndRow = cnvStartRow + static_cast<int>(cnvRows); // end index + 1

    // create input and output image accessors
    // and advance output accessor to lower left pixel that is set by convolution
    InPixelAccessor mImageRowAcc(maskedImage);
    OutPixelAccessor cnvRowAcc(convolvedImage);
    cnvRowAcc.advance(cnvStartCol, cnvStartRow);
    
    std::vector<lsst::afw::math::Kernel::PixelT> kColVec(kernel.getCols());
    std::vector<lsst::afw::math::Kernel::PixelT> kRowVec(kernel.getRows());
    
    if (kernel.isSpatiallyVarying()) {
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially varying separable kernel");
        for (int cnvRow = cnvStartRow; cnvRow < cnvEndRow; ++cnvRow, cnvRowAcc.nextRow(), mImageRowAcc.nextRow()) {
            double rowPos = lsst::afw::image::indexToPosition(cnvRow);
            InPixelAccessor mImageColAcc = mImageRowAcc;
            OutPixelAccessor cnvColAcc = cnvRowAcc;
            for (int cnvCol = cnvStartCol; cnvCol < cnvEndCol; ++cnvCol, mImageColAcc.nextCol(), cnvColAcc.nextCol()) {
                double colPos = lsst::afw::image::indexToPosition(cnvCol);
                KernelPixelT kSum;
                kernel.computeVectors(kColVec, kRowVec, kSum, doNormalize, colPos, rowPos);
                // g++ 3.6.4 requires the template arguments here to find the function; I don't know why
                // Is this still true? RHL
                lsst::afw::math::apply<OutPixelT, InPixelT, MaskPixelT>(cnvColAcc, mImageColAcc, kColVec, kRowVec);
                if (doNormalize) {
                    *(cnvColAcc.image) /= static_cast<OutPixelT>(kSum);
                    *(cnvColAcc.variance) /= static_cast<OutPixelT>(kSum * kSum);
                }
            }
        }
    } else {
        // kernel is spatially invariant
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially invariant separable kernel");
        KernelPixelT kSum;
        kernel.computeVectors(kColVec, kRowVec, kSum, doNormalize);
        for (int cnvRow = cnvStartRow; cnvRow < cnvEndRow; ++cnvRow, cnvRowAcc.nextRow(), mImageRowAcc.nextRow()) {
            InPixelAccessor mImageColAcc = mImageRowAcc;
            OutPixelAccessor cnvColAcc = cnvRowAcc;
            for (int cnvCol = cnvStartCol; cnvCol < cnvEndCol; ++cnvCol, mImageColAcc.nextCol(), cnvColAcc.nextCol()) {
                // g++ 3.6.4 requires the template arguments here to find the function; I don't know why
                lsst::afw::math::apply<OutPixelT, InPixelT, MaskPixelT>(cnvColAcc, mImageColAcc, kColVec, kRowVec);
            }
        }
    }
}

/**
 * @brief Convolve a MaskedImage with a Kernel, setting pixels of an existing image
 *
 * convolvedImage must be the same size as maskedImage.
 * convolvedImage has a border in which the output pixels are just a copy of the input pixels
 * and the output mask bit edgeBit has been set. This border has size:
 * * kernel.getCtrCol/Row() along the left/bottom edge
 * * kernel.getCols/Rows() - 1 - kernel.getCtrCol/Row() along the right/top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameter if convolvedImage is not the same size as maskedImage.
 * @throw lsst::pex::exceptions::InvalidParameter if maskedImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename OutPixelT, typename InPixelT, typename MaskPixelT, typename KernelT>
void lsst::afw::math::convolve(
    lsst::afw::image::MaskedImage<OutPixelT, MaskPixelT> &convolvedImage,       ///< convolved image
    lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> const &maskedImage,    ///< image to convolve
    KernelT const &kernel,    ///< convolution kernel
    int edgeBit,        ///< mask bit to indicate pixel includes edge-extended data;
                        ///< if negative then no bit is set
    bool doNormalize    ///< if True, normalize the kernel, else use "as is"
) {
    lsst::afw::math::basicConvolve(convolvedImage, maskedImage, kernel, doNormalize);
    _copyBorder(convolvedImage, maskedImage, kernel, edgeBit);
}

/**
 * @brief Convolve a MaskedImage with a Kernel, returning a new image.
 *
 * @return the convolved MaskedImage.
 *
 * The returned MaskedImage is the same size as maskedImage.
 * It has a border in which the output pixels are just a copy of the input pixels
 * and the output mask bit edgeBit has been set. This border will have size:
 * * kernel.getCtrCol/Row() along the left/bottom edge
 * * kernel.getCols/Rows() - 1 - kernel.getCtrCol/Row() along the right/top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameter if maskedImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename InPixelT, typename MaskPixelT, typename KernelT>
lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> lsst::afw::math::convolveNew(
    lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> const &maskedImage,    ///< image to convolve
    KernelT const &kernel,              ///< convolution kernel
    int edgeBit,        ///< mask bit to indicate pixel includes edge-extended data;
                        ///< if negative then no bit is set
    bool doNormalize    ///< if True, normalize the kernel, else use "as is"
) {
    lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> convolvedImage(maskedImage.getCols(), maskedImage.getRows());
    lsst::afw::math::convolve(convolvedImage, maskedImage, kernel, edgeBit, doNormalize);
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
 * @throw lsst::pex::exceptions::InvalidParameter if convolvedImage is not the same size as maskedImage.
 * @throw lsst::pex::exceptions::InvalidParameter if maskedImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename OutPixelT, typename InPixelT, typename MaskPixelT>
void lsst::afw::math::convolveLinear(
    lsst::afw::image::MaskedImage<OutPixelT, MaskPixelT> &convolvedImage,       ///< convolved image
    lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> const &maskedImage,    ///< image to convolve
    lsst::afw::math::LinearCombinationKernel const &kernel,    ///< convolution kernel
    int edgeBit         ///< mask bit to indicate pixel includes edge-extended data;
                        ///< if negative then no bit is set
) {
    if (!kernel.isSpatiallyVarying()) {
        return lsst::afw::math::convolve(convolvedImage, maskedImage, kernel, edgeBit, false);
    }

    const unsigned int imCols = maskedImage.getCols();
    const unsigned int imRows = maskedImage.getRows();
    const unsigned int kCols = kernel.getCols();
    const unsigned int kRows = kernel.getRows();
    if ((convolvedImage.getCols() != imCols) || (convolvedImage.getRows() != imRows)) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as maskedImage");
    }
    if ((imCols< kCols) || (imRows < kRows)) {
        throw lsst::pex::exceptions::InvalidParameter(
            "maskedImage smaller than kernel in columns and/or rows");
    }
    
    typedef lsst::afw::image::MaskedImage<double, MaskPixelT> BasisMaskedImage;
    typedef boost::shared_ptr<BasisMaskedImage> BasisMaskedImagePtr;
    typedef lsst::afw::image::MaskedPixelAccessor<double, MaskPixelT> BasisAccessor;
    typedef std::vector<BasisAccessor> BasisAccessorListType;
    typedef lsst::afw::image::MaskedPixelAccessor<OutPixelT, MaskPixelT> OutAccessor;
    typedef lsst::afw::math::LinearCombinationKernel::KernelList KernelList;
    typedef std::vector<double> kernelCoeffListType;

    const int cnvCols = static_cast<int>(imCols) + 1 - static_cast<int>(kernel.getCols());
    const int cnvRows = static_cast<int>(imRows) + 1 - static_cast<int>(kernel.getRows());
    const int cnvStartCol = static_cast<int>(kernel.getCtrCol());
    const int cnvStartRow = static_cast<int>(kernel.getCtrRow());
    const int cnvEndCol = cnvStartCol + static_cast<int>(cnvCols); // end index + 1
    const int cnvEndRow = cnvStartRow + static_cast<int>(cnvRows); // end index + 1

    // create a vector of pointers to basis images (input MaskedImage convolved with each basis kernel)
    // and accessors to those images. We don't use the basis images directly,
    // but keep them around to prevent the memory from being reclaimed.
    // Advance the accessors to cnvStart
    KernelList basisKernelList = kernel.getKernelList();
    std::vector<BasisMaskedImagePtr> basisImagePtrList;
    BasisAccessorListType basisImRowAccList;
    for (typename KernelList::const_iterator basisKernelIter = basisKernelList.begin();
        basisKernelIter != basisKernelList.end(); ++basisKernelIter) {
        BasisMaskedImagePtr basisImagePtr(new BasisMaskedImage(imCols, imRows));
        lsst::afw::math::basicConvolve(*basisImagePtr, maskedImage, **basisKernelIter, false);
        basisImagePtrList.push_back(basisImagePtr);
        basisImRowAccList.push_back(lsst::afw::image::MaskedPixelAccessor<double, MaskPixelT>(*basisImagePtr));
        basisImRowAccList.back().advance(cnvStartCol, cnvStartRow);
    }
    
    // iterate over matching pixels of all images to compute output image
    std::vector<double> kernelCoeffList(kernel.getNKernelParameters());
    OutAccessor cnvRowAcc(convolvedImage);
    cnvRowAcc.advance(cnvStartCol, cnvStartRow);
    for (int cnvRow = cnvStartRow; cnvRow < cnvEndRow; ++cnvRow) {
        double rowPos = lsst::afw::image::indexToPosition(cnvRow);
        OutAccessor cnvColAcc = cnvRowAcc;
        BasisAccessorListType basisImColAccList = basisImRowAccList;
        for (int cnvCol = cnvStartCol; cnvCol < cnvEndCol; ++cnvCol) {
            double colPos = lsst::afw::image::indexToPosition(cnvCol);
            kernel.computeKernelParametersFromSpatialModel(kernelCoeffList, colPos, rowPos);

            double cnvImagePix = 0.0;
            double cnvNoisePix = 0.0;
            *(cnvColAcc.mask) = 0;
            typename std::vector<double>::const_iterator kernelCoeffIter = kernelCoeffList.begin();
            for (typename BasisAccessorListType::const_iterator basisImColIter = basisImColAccList.begin();
                basisImColIter != basisImColAccList.end(); ++basisImColIter, ++kernelCoeffIter) {
                cnvImagePix += (*kernelCoeffIter) * (*(basisImColIter->image));
                cnvNoisePix += (*kernelCoeffIter) * std::sqrt(*(basisImColIter->variance));
                *(cnvColAcc.mask) |= *(basisImColIter->mask);
            }
            *(cnvColAcc.image) = static_cast<OutPixelT>(cnvImagePix);
            *(cnvColAcc.variance) = static_cast<OutPixelT>(cnvNoisePix * cnvNoisePix);
            cnvColAcc.nextCol();
            std::for_each(basisImColAccList.begin(), basisImColAccList.end(),
                std::mem_fun_ref(&BasisAccessor::nextCol));
        }
        cnvRowAcc.nextRow();
        std::for_each(basisImRowAccList.begin(), basisImRowAccList.end(),
            std::mem_fun_ref(&BasisAccessor::nextRow));
    }

    _copyBorder(convolvedImage, maskedImage, kernel, edgeBit);
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
 * @throw lsst::pex::exceptions::InvalidParameter if maskedImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename InPixelT, typename MaskPixelT>
lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> lsst::afw::math::convolveLinearNew(
    lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> const &maskedImage,    ///< image to convolve
    lsst::afw::math::LinearCombinationKernel const &kernel,    ///< convolution kernel
    int edgeBit         ///< mask bit to indicate pixel includes edge-extended data;
                        ///< if negative then no bit is set
) {
    lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> convolvedImage(maskedImage.dimensions());
    lsst::afw::math::convolveLinear(convolvedImage, maskedImage, kernel, edgeBit);
    return convolvedImage;
}
