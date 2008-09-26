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

// declare private functions
template <typename OutPixelT, typename InPixelT>
void _copyBorder(
    lsst::afw::image::Image<OutPixelT> &convolvedImage,
    lsst::afw::image::Image<InPixelT> const &inImage,
    lsst::afw::math::Kernel const &kernel
);

template <typename OutPixelT, typename InPixelT>
inline void _copyRegion(
    typename lsst::afw::image::Image<OutPixelT> &outImage,
    typename lsst::afw::image::Image<InPixelT> const &inImage,
    vw::BBox2i const &region
);

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
    typename lsst::afw::image::Image<InPixelT>::pixel_accessor const &imageAccessor,
        ///< accessor to for image pixel that overlaps (0,0) pixel of kernel(!)
    lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::pixel_accessor const &kernelAccessor,
        ///< accessor for (0,0) pixel of kernel
    unsigned int cols,  ///< number of columns in kernel
    unsigned int rows   ///< number of rows in kernel
) {
    typedef typename lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::pixel_accessor kernelAccessorType;
    double outImage = 0;
    typename lsst::afw::image::Image<InPixelT>::pixel_accessor imageRowAcc = imageAccessor;
    kernelAccessorType kRow = kernelAccessor;
    for (unsigned int row = 0; row < rows; ++row, imageRowAcc.next_row(), kRow.next_row()) {
        InPixelT *imagePtr = &(*imageRowAcc);
        // assume data contiguous along rows; use a pointer instead of a vw pixel accessor to gain speed
        lsst::afw::math::Kernel::PixelT *kerPtr = &(*kRow);
        for (unsigned int col = 0; col < cols; ++col, imagePtr++, kerPtr++) {
            lsst::afw::math::Kernel::PixelT ker = *kerPtr;

            outImage += static_cast<OutPixelT>(ker * (*imagePtr));
        }
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
    typename lsst::afw::image::Image<InPixelT>::pixel_accessor const &imageAccessor,
        ///< accessor to for image pixel that overlaps (0,0) pixel of kernel(!)
    std::vector<lsst::afw::math::Kernel::PixelT> const &kernelColList,  ///< kernel column vector
    std::vector<lsst::afw::math::Kernel::PixelT> const &kernelRowList   ///< kernel row vector
) {
    double outImage = 0;
    typename lsst::afw::image::Image<InPixelT>::pixel_accessor imageRowAcc = imageAccessor;
    std::vector<lsst::afw::math::Kernel::PixelT>::const_iterator kernelRowIter = kernelRowList.begin();
    for ( ; kernelRowIter != kernelRowList.end(); ++kernelRowIter, imageRowAcc.next_row()) {
        typename lsst::afw::image::Image<InPixelT>::pixel_accessor imageColAcc = imageRowAcc;
        std::vector<lsst::afw::math::Kernel::PixelT>::const_iterator kernelColIter = kernelColList.begin();
        double outImageRow = 0;
        for ( ; kernelColIter != kernelColList.end(); ++kernelColIter, imageColAcc.next_col()) {
            double kernelColValue = static_cast<double> (*kernelColIter);
            outImageRow += kernelColValue * (*imageColAcc);
        }
        double kernelRowValue = static_cast<double> (*kernelRowIter);
        outImage += kernelRowValue * outImageRow;
    }
    outValue = static_cast<OutPixelT>(outImage);
}

/**
 * @brief Low-level convolution function that does not set edge pixels.
 *
 * convolvedImage must be the same size as inImage.
 * convolvedImage has a border in which the output pixels are not set. This border has size:
 * * kernel.getCtrCol/Row() along the left/bottom edge
 * * kernel.getCols/Rows() - 1 - kernel.getCtrCol/Row() along the right/top edge
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
    typedef typename lsst::afw::image::Image<InPixelT>::pixel_accessor InPixelAccessor;
    typedef typename lsst::afw::image::Image<OutPixelT>::pixel_accessor OutPixelAccessor;
    typedef typename lsst::afw::image::Image<KernelPixelT>::pixel_accessor KernelAccessor;

    const unsigned int inImageCols = inImage.getCols();
    const unsigned int inImageRows = inImage.getRows();
    const unsigned int kCols = kernel.getCols();
    const unsigned int kRows = kernel.getRows();
    if ((convolvedImage.getCols() != inImageCols) || (convolvedImage.getRows() != inImageRows)) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if ((inImageCols < kCols) || (inImageRows < kRows)) {
        throw lsst::pex::exceptions::InvalidParameter("inImage smaller than kernel in columns and/or rows");
    }
    
    const int cnvCols = static_cast<int>(inImageCols) + 1 - static_cast<int>(kernel.getCols());
    const int cnvRows = static_cast<int>(inImageRows) + 1 - static_cast<int>(kernel.getRows());
    const int cnvStartCol = static_cast<int>(kernel.getCtrCol());
    const int cnvStartRow = static_cast<int>(kernel.getCtrRow());
    const int cnvEndCol = cnvStartCol + static_cast<int>(cnvCols); // end index + 1
    const int cnvEndRow = cnvStartRow + static_cast<int>(cnvRows); // end index + 1

    // create input and output image accessors
    // and advance output accessor to lower left pixel that is set by convolution
    InPixelAccessor inImageRowAcc = inImage.origin();
    OutPixelAccessor cnvImageRowAcc = convolvedImage.origin();
    cnvImageRowAcc.advance(cnvStartCol, cnvStartRow);
    
    if (kernel.isSpatiallyVarying()) {
        lsst::afw::image::Image<KernelPixelT> kernelImage(kernel.getCols(), kernel.getRows());
        KernelAccessor kernelAcc = kernelImage.origin();
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is spatially varying");
        for (int cnvRow = cnvStartRow; cnvRow < cnvEndRow;
            ++cnvRow, cnvImageRowAcc.next_row(), inImageRowAcc.next_row()) {
            double rowPos = lsst::afw::image::indexToPosition(cnvRow);
            InPixelAccessor inImageColAcc = inImageRowAcc;
            OutPixelAccessor cnvImageColAcc = cnvImageRowAcc;
            for (int cnvCol = cnvStartCol; cnvCol < cnvEndCol;
                ++cnvCol, inImageColAcc.next_col(), cnvImageColAcc.next_col()) {
                double colPos = lsst::afw::image::indexToPosition(cnvCol);
                KernelPixelT kSum;
                kernel.computeImage(kernelImage, kSum, false, colPos, rowPos);
                // g++ 3.6.4 requires the template arguments here to find the function; I don't know why
                // Is this still true? RHL
                lsst::afw::math::apply<OutPixelT, InPixelT>(
                    *cnvImageColAcc, inImageColAcc, kernelAcc, kCols, kRows);
                if (doNormalize) {
                    *cnvImageColAcc /= static_cast<InPixelT>(kSum);
                }
            }
        }
    } else {
        // kernel is spatially invariant
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is spatially invariant");
        KernelPixelT kSum;
        lsst::afw::image::Image<KernelPixelT> kernelImage = kernel.computeNewImage(kSum, doNormalize, 0.0, 0.0);
        KernelAccessor kernelAcc = kernelImage.origin();
        for (int cnvRow = cnvStartRow; cnvRow < cnvEndRow;
            ++cnvRow, cnvImageRowAcc.next_row(), inImageRowAcc.next_row()) {
            InPixelAccessor inImageColAcc = inImageRowAcc;
            OutPixelAccessor cnvImageColAcc = cnvImageRowAcc;
            for (int cnvCol = cnvStartCol; cnvCol < cnvEndCol;
                ++cnvCol, inImageColAcc.next_col(), cnvImageColAcc.next_col()) {
                // g++ 3.6.4 requires the template arguments here to find the function; I don't know why
                lsst::afw::math::apply<OutPixelT, InPixelT>(
                    *cnvImageColAcc, inImageColAcc, kernelAcc, kCols, kRows);
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

    const unsigned int inImageCols = inImage.getCols();
    const unsigned int inImageRows = inImage.getRows();
    const unsigned int kCols = kernel.getCols();
    const unsigned int kRows = kernel.getRows();

    if (convolvedImage.getCols() != inImageCols || convolvedImage.getRows() != inImageRows) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if ((inImageCols < kCols) || (inImageRows < kRows)) {
        throw lsst::pex::exceptions::InvalidParameter("inImage smaller than kernel in columns and/or rows");
    }
    
    const int cnvCols = static_cast<int>(inImageCols) + 1 - static_cast<int>(kernel.getCols());
    const int cnvRows = static_cast<int>(inImageRows) + 1 - static_cast<int>(kernel.getRows());
    const int cnvStartCol = static_cast<int>(kernel.getCtrCol());
    const int cnvStartRow = static_cast<int>(kernel.getCtrRow());
    const int inStartCol = kernel.getPixel().first;
    const int inStartRow = kernel.getPixel().second;

    // create input and output image accessors
    // and advance each to the right spot
    InPixelAccessor inImageRowAcc = inImage.origin();
    inImageRowAcc.advance(inStartCol, inStartRow);
    OutPixelAccessor cnvImageRowAcc = convolvedImage.origin();
    cnvImageRowAcc.advance(cnvStartCol, cnvStartRow);

    lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially invariant delta function basis");
    for (int i = 0; i < cnvRows; ++i, cnvImageRowAcc.next_row(), inImageRowAcc.next_row()) {
        InPixelAccessor inImageColAcc = inImageRowAcc;
        OutPixelAccessor cnvImageColAcc = cnvImageRowAcc;
        for (int j = 0; j < cnvCols; ++j, inImageColAcc.next_col(), cnvImageColAcc.next_col()) {
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

    const unsigned int inImageCols = inImage.getCols();
    const unsigned int inImageRows = inImage.getRows();
    const unsigned int kCols = kernel.getCols();
    const unsigned int kRows = kernel.getRows();
    if ((convolvedImage.getCols() != inImageCols) || (convolvedImage.getRows() != inImageRows)) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if ((inImageCols < kCols) || (inImageRows < kRows)) {
        throw lsst::pex::exceptions::InvalidParameter("inImage smaller than kernel in columns and/or rows");
    }
    
    const int cnvCols = static_cast<int>(inImageCols) + 1 - static_cast<int>(kernel.getCols());
    const int cnvRows = static_cast<int>(inImageRows) + 1 - static_cast<int>(kernel.getRows());
    const int cnvStartCol = static_cast<int>(kernel.getCtrCol());
    const int cnvStartRow = static_cast<int>(kernel.getCtrRow());
    const int cnvEndCol = cnvStartCol + static_cast<int>(cnvCols); // end index + 1
    const int cnvEndRow = cnvStartRow + static_cast<int>(cnvRows); // end index + 1

    // create input and output image accessors
    // and advance output accessor to lower left pixel that is set by convolution
    InPixelAccessor inImageRowAcc = inImage.origin();
    OutPixelAccessor cnvImageRowAcc = convolvedImage.origin();
    cnvImageRowAcc.advance(cnvStartCol, cnvStartRow);
    
    std::vector<lsst::afw::math::Kernel::PixelT> kColVec(kernel.getCols());
    std::vector<lsst::afw::math::Kernel::PixelT> kRowVec(kernel.getRows());
    
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
 * * kernel.getCtrCol/Row() along the left/bottom edge
 * * kernel.getCols/Rows() - 1 - kernel.getCtrCol/Row() along the right/top edge
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
 * * kernel.getCtrCol/Row() along the left/bottom edge
 * * kernel.getCols/Rows() - 1 - kernel.getCtrCol/Row() along the right/top edge
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
    lsst::afw::image::Image<InPixelT> convolvedImage(inImage.getCols(), inImage.getRows());
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

    const unsigned int imCols = inImage.getCols();
    const unsigned int imRows = inImage.getRows();
    const unsigned int kCols = kernel.getCols();
    const unsigned int kRows = kernel.getRows();
    if ((convolvedImage.getCols() != imCols) || (convolvedImage.getRows() != imRows)) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if ((imCols< kCols) || (imRows < kRows)) {
        throw lsst::pex::exceptions::InvalidParameter(
            "inImage smaller than kernel in columns and/or rows");
    }
    
    typedef lsst::afw::image::Image<double> BasisImage;
    typedef boost::shared_ptr<BasisImage> BasisImagePtr;
    typedef typename lsst::afw::image::Image<double>::pixel_accessor BasisPixelAccessor;
    typedef std::vector<BasisPixelAccessor> BasisPixelAccessorList;
    typedef typename lsst::afw::image::Image<OutPixelT>::pixel_accessor OutPixelAccessor;
    typedef lsst::afw::math::LinearCombinationKernel::KernelList kernelListType;
    typedef std::vector<double> kernelCoeffListType;

    const int cnvCols = static_cast<int>(imCols) + 1 - static_cast<int>(kernel.getCols());
    const int cnvRows = static_cast<int>(imRows) + 1 - static_cast<int>(kernel.getRows());
    const int cnvStartCol = static_cast<int>(kernel.getCtrCol());
    const int cnvStartRow = static_cast<int>(kernel.getCtrRow());
    const int cnvEndCol = cnvStartCol + static_cast<int>(cnvCols); // end index + 1
    const int cnvEndRow = cnvStartRow + static_cast<int>(cnvRows); // end index + 1

    // create a vector of pointers to basis images (input Image convolved with each basis kernel)
    // and accessors to those images. We don't use the basis images directly,
    // but keep them around to prevent the memory from being reclaimed.
    // Advance the accessors to cnvStart
    kernelListType basisKernelList = kernel.getKernelList();
    std::vector<BasisImagePtr> basisImagePtrList;
    BasisPixelAccessorList basisImRowAccList;
    for (typename kernelListType::const_iterator basisKernelIter = basisKernelList.begin();
        basisKernelIter != basisKernelList.end(); ++basisKernelIter) {
        BasisImagePtr basisImagePtr(new BasisImage(imCols, imRows));
        lsst::afw::math::basicConvolve(*basisImagePtr, inImage, **basisKernelIter, false);
        basisImagePtrList.push_back(basisImagePtr);
        basisImRowAccList.push_back(BasisPixelAccessor(basisImagePtr->origin()));
        basisImRowAccList.back().advance(cnvStartCol, cnvStartRow);
    }
    
    // iterate over matching pixels of all images to compute output image
    std::vector<double> kernelCoeffList(kernel.getNKernelParameters());
    OutPixelAccessor cnvRowAcc = convolvedImage.origin();
    cnvRowAcc.advance(cnvStartCol, cnvStartRow);
    for (int cnvRow = cnvStartRow; cnvRow < cnvEndRow; ++cnvRow) {
        double rowPos = lsst::afw::image::indexToPosition(cnvRow);
        OutPixelAccessor cnvColAcc = cnvRowAcc;
        BasisPixelAccessorList basisImColAccList = basisImRowAccList;
        for (int cnvCol = cnvStartCol; cnvCol < cnvEndCol; ++cnvCol) {
            double colPos = lsst::afw::image::indexToPosition(cnvCol);
            kernel.computeKernelParametersFromSpatialModel(kernelCoeffList, colPos, rowPos);

            double cnvImagePix = 0.0;
            typename std::vector<double>::const_iterator kernelCoeffIter = kernelCoeffList.begin();
            for (typename BasisPixelAccessorList::const_iterator basisImColIter = basisImColAccList.begin();
                basisImColIter != basisImColAccList.end(); ++basisImColIter, ++kernelCoeffIter) {
                cnvImagePix += (*kernelCoeffIter) * (**basisImColIter);
            }
            *cnvColAcc = static_cast<OutPixelT>(cnvImagePix);
            cnvColAcc.next_col();
            std::for_each(basisImColAccList.begin(), basisImColAccList.end(),
                std::mem_fun_ref(&BasisPixelAccessor::next_col));
        }
        cnvRowAcc.next_row();
        std::for_each(basisImRowAccList.begin(), basisImRowAccList.end(),
            std::mem_fun_ref(&BasisPixelAccessor::next_row));
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
    lsst::afw::image::Image<InPixelT> convolvedImage(inImage.getCols(), inImage.getRows());
    lsst::afw::math::convolveLinear(convolvedImage, inImage, kernel);
    return convolvedImage;
}

/**
 * @brief Private function to copy the border of a convolved image.
 *
 * Copy the border of an image. This border has size:
 * * kernel.getCtrCol/Row() along the left/bottom edge
 * * kernel.getCols/Rows() - 1 - kernel.getCtrCol/Row() along the right/top edge
 *
 * The sizes are not error-checked.
 *
 * @ingroup afw
 */
template <typename OutPixelT, typename InPixelT>
void _copyBorder(
    lsst::afw::image::Image<OutPixelT> &convolvedImage,       ///< convolved image
    lsst::afw::image::Image<InPixelT> const &inImage,    ///< image to convolve
    lsst::afw::math::Kernel const &kernel   ///< convolution kernel
) {
    const unsigned int imCols = inImage.getCols();
    const unsigned int imRows = inImage.getRows();
    const unsigned int kCols = kernel.getCols();
    const unsigned int kRows = kernel.getRows();
    const unsigned int kCtrCol = kernel.getCtrCol();
    const unsigned int kCtrRow = kernel.getCtrRow();
    
    vw::BBox2i bottomEdge(0, 0, imCols, kCtrRow);
    _copyRegion(convolvedImage, inImage, bottomEdge);
    
    vw::int32 numRows = kRows - (1 + kCtrRow);
    vw::BBox2i topEdge(0, imRows - numRows, imCols, numRows);
    _copyRegion(convolvedImage, inImage, topEdge);

    vw::BBox2i leftEdge(0, kCtrRow, kCtrCol, imRows + 1 - kRows);
    _copyRegion(convolvedImage, inImage, leftEdge);
    
    vw::int32 numCols = kCols - (1 + kernel.getCtrCol());
    vw::BBox2i rightEdge(imCols - numCols, kCtrRow, numCols, imRows + 1 - kRows);
    _copyRegion(convolvedImage, inImage, rightEdge);
}


/**
 * @brief Private function to copy a rectangular region from one Image to another.
 *
 * I hope eventually to replace this by calls to Image.getSubImage
 * and Image.replaceSubImage, but that is currently too messy
 * because getSubImage requires a shared pointer to the source image.
 *
 * @throw invalid_argument if the region extends off of either image.
 */
template <typename OutPixelT, typename InPixelT>
inline void _copyRegion(
    typename lsst::afw::image::Image<OutPixelT> &outImage,          ///< destination Image
    typename lsst::afw::image::Image<InPixelT> const &inImage,   ///< source Image
    vw::BBox2i const &region    ///< region to copy
) {
    typedef typename lsst::afw::image::Image<InPixelT>::pixel_accessor InPixelAccessor;
    typedef typename lsst::afw::image::Image<OutPixelT>::pixel_accessor OutPixelAccessor;

    vw::math::Vector<vw::int32> const startColRow = region.min();
    vw::math::Vector<vw::int32> const numColRow = region.size();
    lsst::pex::logging::Trace("lsst.afw.kernel._copyRegion", 4,
        "_copyRegion: dest size=%d, %d; src size=%d, %d; region start=%d, %d; region size=%d, %d",
        outImage.getCols(), outImage.getRows(), inImage.getCols(), inImage.getRows(),
        startColRow[0], startColRow[1], numColRow[0], numColRow[1]
    );

    vw::math::Vector<vw::int32> const endColRow = region.max();
    if ((static_cast<unsigned int>(endColRow[0]) > std::min(outImage.getCols(), inImage.getCols()))
        || ((static_cast<unsigned int>(endColRow[1]) > std::min(outImage.getRows(), inImage.getRows())))) {
        throw lsst::pex::exceptions::InvalidParameter("Region out of range");
    }
    InPixelAccessor inRow = inImage.origin();
    OutPixelAccessor outRow = outImage.origin();
    inRow.advance(startColRow[0], startColRow[1]);
    outRow.advance(startColRow[0], startColRow[1]);
    for (int row = 0; row < numColRow[1]; ++row, inRow.next_row(), outRow.next_row()) {
        InPixelAccessor inCol = inRow;
        OutPixelAccessor outCol = outRow;
        for (int col = 0; col < numColRow[0]; ++col, inCol.next_col(), outCol.next_col()) {
            *outCol = static_cast<OutPixelT>(*inCol);
        }
    }
}
