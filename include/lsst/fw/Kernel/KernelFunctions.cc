// -*- LSST-C++ -*-
/**
 * \file
 *
 * \brief Definition of templated functions declared in KernelFunctions.h
 *
 * This file is meant to be included by lsst/fw/KernelFunctions.h
 *
 * \author Russell Owen
 *
 * \ingroup fw
 */
#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

#include <boost/format.hpp>
#include <vw/Image.h>

#include <lsst/mwi/utils/Trace.h>
#include <lsst/fw/ImageUtils.h>

/**
 * \brief Apply convolution kernel to a masked image at one point
 *
 * Note: this is a high performance routine; the user is expected to:
 * - handle edge extension
 * - figure out the kernel center and adjust the supplied pixel accessors accordingly
 * For an example of how to do this see the convolve function.
 *
 * \ingroup fw
 */
template <typename ImageT, typename MaskT, typename KernelT>
inline void lsst::fw::kernel::apply(
    lsst::fw::MaskedPixelAccessor<ImageT, MaskT> &outAccessor,    ///< accessor for output pixel
    lsst::fw::MaskedPixelAccessor<ImageT, MaskT> const &imageAccessor,
        ///< accessor to for image pixel that overlaps (0,0) pixel of kernel(!)
    typename lsst::fw::Image<KernelT>::pixel_accessor const &kernelAccessor,
        ///< accessor for (0,0) pixel of kernel
    unsigned int cols,      ///< number of columns in kernel
    unsigned int rows,      ///< number of rows in kernel
    KernelT threshold   ///< if a kernel pixel > threshold then the corresponding image mask pixel
                        ///< is ORd into the output pixel, otherwise the mask pixel is ignored
) {
    typedef typename lsst::fw::Image<KernelT>::pixel_accessor kernelAccessorType;
    *outAccessor.image = 0;
    *outAccessor.variance = 0;
    *outAccessor.mask = 0;

    lsst::fw::MaskedPixelAccessor<ImageT, MaskT> imRow = imageAccessor;
    kernelAccessorType kRow = kernelAccessor;
    for (unsigned int row = 0; row < rows; ++row, imRow.nextRow(), kRow.next_row()) {
        MaskedPixelAccessor<ImageT, MaskT> imCol = imRow;
        kernelAccessorType kCol = kRow;
        for (unsigned int col = 0; col < cols; ++col, imCol.nextCol(), kCol.next_col()) {
            *outAccessor.image += (*kCol) * (*imCol.image);
            *outAccessor.variance += (*kCol) * (*kCol) * (*imCol.variance);
            if ((*imCol.mask) && (*kCol > threshold)) {
                // this bad pixel contributes enough to "OR" in the bad bits
                *outAccessor.mask |= *imCol.mask;
            }
        }
    }
}

/**
 * \brief Convolve a MaskedImage with a Kernel, setting pixels of an existing image
 *
 * convolvedImage must be the same size as maskedImage.
 * It has a border in which the output pixels are just a copy of the input pixels
 * and the output mask bit edgeBit has been set. This border will have size:
 * * kernel.getCtrCol/Row() along the left/bottom edge
 * * kernel.getCols/Rows() - 1 - kernel.getCtrCol/Row() along the right/top edge
 *
 * \throw lsst::mwi::exceptions::InvalidParameter if convolvedImage is not the same size as maskedImage.
 * \throw lsst::mwi::exceptions::InvalidParameter if maskedImage is smaller (in colums or rows) than kernel.
 *
 * \ingroup fw
 */
template <typename ImageT, typename MaskT, typename KernelT>
void lsst::fw::kernel::convolve(
    lsst::fw::MaskedImage<ImageT, MaskT> &convolvedImage,       ///< convolved image
    lsst::fw::MaskedImage<ImageT, MaskT> const &maskedImage,    ///< image to convolve
    lsst::fw::Kernel<KernelT> const &kernel,    ///< convolution kernel
    KernelT threshold,  ///< if kernel pixel > threshold then corresponding maskedImage mask pixel is OR'd in
    int edgeBit,        ///< mask bit to indicate pixel includes edge-extended data;
                        ///< if negative then no bit is set
    bool doNormalize    ///< if True, normalize the kernel, else use "as is"
) {
    typedef lsst::fw::MaskedPixelAccessor<ImageT, MaskT> imageAccessorType;
    typedef typename lsst::fw::Image<KernelT>::pixel_accessor kernelAccessorType;

    const unsigned int imCols = maskedImage.getCols();
    const unsigned int imRows = maskedImage.getRows();
    const unsigned int kCols = kernel.getCols();
    const unsigned int kRows = kernel.getRows();
    const unsigned int kCtrCol = kernel.getCtrCol();
    const unsigned int kCtrRow = kernel.getCtrRow();
    if ((convolvedImage.getCols() != imCols) || (convolvedImage.getRows() != imRows)) {
        throw lsst::mwi::exceptions::InvalidParameter("convolvedImage not the same size as maskedImage");
    }
    if ((imCols< kCols) || (imRows < kRows)) {
        throw lsst::mwi::exceptions::InvalidParameter(
            "maskedImage smaller than kernel in columns and/or rows");
    }

    const unsigned int cnvCols = imCols + 1 - kernel.getCols();
    const unsigned int cnvRows = imRows + 1 - kernel.getRows();

    // create input and output image accessors
    // and advance output accessor to lower left pixel that is set by convolution
    imageAccessorType imRow(maskedImage);
    imageAccessorType outRow(convolvedImage);
    outRow.advance(kernel.getCtrCol(), kernel.getCtrRow());
    
    if (kernel.isSpatiallyVarying()) {
        lsst::fw::Image<KernelT> kernelImage(kernel.getCols(), kernel.getRows());
        kernelAccessorType kernelAccessor = kernelImage.origin();
        lsst::mwi::utils::Trace("lsst.fw.kernel.convolve", 1, "kernel is spatially varying");
        for (int row = 0; row < static_cast<int>(cnvRows); ++row, outRow.nextRow(), imRow.nextRow()) {
            double rowPos = lsst::fw::image::indexToPosition(row);
            imageAccessorType imCol = imRow;
            imageAccessorType outCol = outRow;
            for (int col = 0; col < static_cast<int>(cnvCols); ++col, imCol.nextCol(), outCol.nextCol()) {
                KernelT kSum;
                kernel.computeImage(
                    kernelImage, kSum, lsst::fw::image::indexToPosition(col), rowPos, false);
                KernelT adjThreshold = threshold * kSum;
                lsst::fw::kernel::apply(outCol, imCol, kernelAccessor, kCols, kRows, adjThreshold);
                if (doNormalize) {
                    *outCol.image /= static_cast<ImageT>(kSum);
                    *outCol.variance /= static_cast<ImageT>(kSum * kSum);
                }
            }
        }
    } else {
        // kernel is spatially invariant
        lsst::mwi::utils::Trace("lsst.fw.kernel.convolve", 1, "kernel is spatially invariant");
        KernelT kSum;
        lsst::fw::Image<KernelT> kernelImage = kernel.computeNewImage(kSum, 0.0, 0.0, doNormalize);
        kernelAccessorType kernelAccessor = kernelImage.origin();
        for (int row = 0; row < static_cast<int>(cnvRows); ++row, outRow.nextRow(), imRow.nextRow()) {
            imageAccessorType imCol = imRow;
            imageAccessorType outCol = outRow;
            for (int col = 0; col < static_cast<int>(cnvCols); ++col, imCol.nextCol(), outCol.nextCol()) {
                lsst::fw::kernel::apply(outCol, imCol, kernelAccessor, kCols, kRows, threshold);
            }
        }
    }

    // set edge pixels
    MaskT edgeOrVal = edgeBit < 0 ? 0 : 1 << edgeBit;
    
    vw::BBox2i bottomEdge(0, 0, imCols, kCtrRow);
    lsst::fw::kernel::_copyRegion(convolvedImage, maskedImage, bottomEdge, edgeOrVal);
    
    vw::int32 numRows = kCols - (1 + kCtrRow);
    vw::BBox2i topEdge(0, imRows - numRows, imCols, numRows);
    lsst::fw::kernel::_copyRegion(convolvedImage, maskedImage, topEdge, edgeOrVal);

    vw::BBox2i leftEdge(0, kCtrRow, kCtrCol, imRows + 1 - kRows);
    lsst::fw::kernel::_copyRegion(convolvedImage, maskedImage, leftEdge, edgeOrVal);
    
    vw::int32 numCols = kCols - (1 + kernel.getCtrCol());
    vw::BBox2i rightEdge(imCols - numCols, kCtrRow, numCols, imRows + 1 - kRows);
    lsst::fw::kernel::_copyRegion(convolvedImage, maskedImage, rightEdge, edgeOrVal);
}

/**
 * \brief Convolve a MaskedImage with a Kernel, returning a new image.
 *
 * \return the convolved MaskedImage.
 *
 * The returned MaskedImage is the same size as maskedImage.
 * It has a border in which the output pixels are just a copy of the input pixels
 * and the output mask bit edgeBit has been set. This border will have size:
 * * kernel.getCtrCol/Row() along the left/bottom edge
 * * kernel.getCols/Rows() - 1 - kernel.getCtrCol/Row() along the right/top edge
 *
 * \throw lsst::mwi::exceptions::InvalidParameter if maskedImage is smaller (in colums or rows) than kernel.
 *
 * \ingroup fw
 */
template <typename ImageT, typename MaskT, typename KernelT>
lsst::fw::MaskedImage<ImageT, MaskT> lsst::fw::kernel::convolve(
    lsst::fw::MaskedImage<ImageT, MaskT> const &maskedImage,    ///< image to convolve
    lsst::fw::Kernel<KernelT> const &kernel,    ///< convolution kernel
    KernelT threshold,  ///< if kernel pixel > threshold then corresponding maskedImage mask pixel is OR'd in
    int edgeBit,        ///< mask bit to indicate pixel includes edge-extended data;
                        ///< if negative then no bit is set
    bool doNormalize    ///< if True, normalize the kernel, else use "as is"
) {
    lsst::fw::MaskedImage<ImageT, MaskT> convolvedImage(maskedImage.getCols(), maskedImage.getRows());
    lsst::fw::kernel::convolve(convolvedImage, maskedImage, kernel, threshold, edgeBit, doNormalize);
    return convolvedImage;
}


/**
 * \brief Convolve a MaskedImage with a LinearCombinationKernel, setting pixels of an existing image
 *
 * A variant of the convolve function that is faster for spatially varying LinearCombinationKernels.
 * For the sake of speed: 
 * * The threshold is fixed at 0
 * * The kernel is NOT normalized
 * If you want a higher threshold or normalization then call the standard convolve function.
 *
 * The Algorithm:
 * Convolves the input MaskedImage by each basis kernel in turn, creating a set of basis images.
 * Then for each output pixel it solves the spatial model and computes the the pixel as
 * the appropriate linear combination of basis images.
 *
 * \throw lsst::mwi::exceptions::InvalidParameter if convolvedImage is not the same size as maskedImage.
 * \throw lsst::mwi::exceptions::InvalidParameter if maskedImage is smaller (in colums or rows) than kernel.
 *
 * \ingroup fw
 */
template <typename ImageT, typename MaskT, typename KernelT>
void lsst::fw::kernel::convolveLinear(
    lsst::fw::MaskedImage<ImageT, MaskT> &convolvedImage,       ///< convolved image
    lsst::fw::MaskedImage<ImageT, MaskT> const &maskedImage,    ///< image to convolve
    lsst::fw::LinearCombinationKernel<KernelT> const &kernel,    ///< convolution kernel
    int edgeBit         ///< mask bit to indicate pixel includes edge-extended data;
                        ///< if negative then no bit is set
) {
    KernelT threshold = 0;

    if (!kernel.isSpatiallyVarying()) {
        return lsst::fw::kernel::convolve(convolvedImage, maskedImage, kernel, threshold, edgeBit, false);
    }

    const unsigned int imCols = maskedImage.getCols();
    const unsigned int imRows = maskedImage.getRows();
    const unsigned int kCols = kernel.getCols();
    const unsigned int kRows = kernel.getRows();
    if ((convolvedImage.getCols() != imCols) || (convolvedImage.getRows() != imRows)) {
        throw lsst::mwi::exceptions::InvalidParameter("convolvedImage not the same size as maskedImage");
    }
    if ((imCols< kCols) || (imRows < kRows)) {
        throw lsst::mwi::exceptions::InvalidParameter(
            "maskedImage smaller than kernel in columns and/or rows");
    }

    typedef lsst::fw::MaskedImage<ImageT, MaskT> maskedImageType;
    typedef lsst::fw::MaskedPixelAccessor<ImageT, MaskT> imageAccessorType;
    imageAccessorType outRow(convolvedImage);
    
    // create vector of basis images (input MaskedImage convolved with each basis kernel)
    // and accessors to those images
    typedef typename lsst::fw::LinearCombinationKernel<KernelT>::KernelListType kernelListType;
    typedef std::vector<maskedImageType> imageListType;
    typedef std::vector<imageAccessorType> imageAccessorListType;
    typedef std::vector<double> kernelCoeffListType;

    kernelListType basisKernels = kernel.getKernelList();
    imageListType basisImageList;
    imageAccessorListType basisImageRowList;
    {
        maskedImageType basisImage(imCols, imRows);
        for (typename kernelListType::const_iterator basisKernelIter = basisKernels.begin();
            basisKernelIter != basisKernels.end(); ++basisKernelIter) {
            lsst::fw::kernel::convolve(basisImage, maskedImage, **basisKernelIter, threshold, edgeBit, false);
            basisImageList.push_back(basisImage);
            basisImageRowList.push_back(lsst::fw::MaskedPixelAccessor<ImageT, MaskT>(basisImage));
        }
    }
    
    // iterate over matching pixels of all images to compute output image
    std::vector<double> kernelCoeffList(kernel.getNKernelParameters());
    for (int row = 0; row < static_cast<int>(imRows); ++row) {
        double rowPos = lsst::fw::image::indexToPosition(row);
        imageAccessorType outCol = outRow;
        imageAccessorListType basisImageColList = basisImageRowList;
        for (int col = 0; col < static_cast<int>(imCols); ++col) {
            double colPos = lsst::fw::image::indexToPosition(col);
            kernel.computeKernelParametersFromSpatialModel(kernelCoeffList, colPos, rowPos);
            
            typename std::vector<double>::const_iterator kernelCoeffIter = kernelCoeffList.begin();
            for (typename imageAccessorListType::const_iterator basisImageColIter = basisImageColList.begin();
                basisImageColIter != basisImageColList.end(); ++basisImageColIter, ++kernelCoeffIter) {
                *outCol.image += (*kernelCoeffIter) * (*basisImageColIter->image);
                *outCol.variance += (*kernelCoeffIter) * (*kernelCoeffIter) * (*basisImageColIter->variance);
                *outCol.mask |= (*basisImageColIter->mask);
            }
            outCol.nextCol();
            std::for_each(basisImageColList.begin(), basisImageColList.end(),
                std::mem_fun_ref(&lsst::fw::MaskedPixelAccessor<ImageT, MaskT>::nextCol));
        }
    }
    outRow.nextRow();
    std::for_each(basisImageRowList.begin(), basisImageRowList.end(),
        std::mem_fun_ref(&lsst::fw::MaskedPixelAccessor<ImageT, MaskT>::nextRow));
}

/**
 * \brief Print the pixel values of a kernel to std::cout
 *
 * Rows increase upward and columns to the right; thus the lower left pixel is (0,0).
 *
 * \ingroup fw
 */
template <typename PixelT>
void lsst::fw::kernel::printKernel(
    lsst::fw::Kernel<PixelT> const &kernel,    ///< the kernel
    double x,   ///< x at which to evaluate kernel
    double y,   ///< y at which to evaluate kernel
    bool doNormalize,   ///< if true, normalize kernel
    std::string pixelFmt     ///< format for pixel values
) {
    typedef typename lsst::fw::Image<PixelT>::pixel_accessor imageAccessorType;
    PixelT kSum;
    lsst::fw::Image<PixelT> kImage = kernel.computeNewImage(kSum, x, y, doNormalize);
    imageAccessorType imRow = kImage.origin();
    imRow.advance(0, kImage.getRows()-1);
    for (unsigned int row=0; row < kImage.getRows(); ++row, imRow.prev_row()) {
        imageAccessorType imCol = imRow;
        for (unsigned int col = 0; col < kImage.getCols(); ++col, imCol.next_col()) {
            std::cout << boost::format(pixelFmt) % (*imCol) << " ";
        }
        std::cout << std::endl;
    }
    if (doNormalize && abs(static_cast<double>(kSum) - 1.0) > 1.0e-5) {
        std::cout << boost::format("Warning! Sum of all pixels = %9.5f != 1.0\n") % kSum;
    }
    std::cout << std::endl;
}

/**
 * \brief Private function to copy a rectangular region from one MaskedImage to another
 *
 * I hope eventually to replace this by calls to MaskedImage.getSubImage
 * and MaskedImage.replaceSubImage, but that is currently too messy
 * because getSubImage requires a shared pointer to the source image.
 *
 * \throw invalid_argument if the region extends off of either image.
 */
template <typename ImageT, typename MaskT>
inline void lsst::fw::kernel::_copyRegion(
    typename lsst::fw::MaskedImage<ImageT, MaskT> &destImage,           ///< destination MaskedImage
    typename lsst::fw::MaskedImage<ImageT, MaskT> const &sourceImage,   ///< source MaskedImage
    vw::BBox2i const &region,   ///< region to copy
    MaskT orMask    ///< data to "or" into the mask pixels
) {
    typedef lsst::fw::MaskedPixelAccessor<ImageT, MaskT> imageAccessorType;

    vw::math::Vector<vw::int32> const startColRow = region.min();
    vw::math::Vector<vw::int32> const numColRow = region.size();
//    lsst::mwi::utils::Trace("lsst.fw.kernel._copyRegion", 1, str(boost::format(
//        "_copyRegion: dest size %d, %d; src size %d, %d; region start=%d, %d; region size=%d, %d; orMask=%d")
//        % destImage.getCols() % destImage.getRows() % sourceImage.getCols() % sourceImage.getRows()
//        % startColRow[0] % startColRow[1]% numColRow[0] % numColRow[1] % orMask
//    ));

    vw::math::Vector<vw::int32> const endColRow = region.max();
    if ((static_cast<unsigned int>(endColRow[0]) > std::min(destImage.getCols(), sourceImage.getCols()))
        || ((static_cast<unsigned int>(endColRow[1]) > std::min(destImage.getRows(), sourceImage.getRows())))) {
        throw lsst::mwi::exceptions::InvalidParameter("Region out of range");
    }
    imageAccessorType inRow(sourceImage);
    imageAccessorType outRow(destImage);
    inRow.advance(startColRow[0], startColRow[1]);
    outRow.advance(startColRow[0], startColRow[1]);
    for (int row = 0; row < numColRow[1]; ++row, inRow.nextRow(), outRow.nextRow()) {
        imageAccessorType inCol = inRow;
        imageAccessorType outCol = outRow;
        for (int col = 0; col < numColRow[0]; ++col, inCol.nextCol(), outCol.nextCol()) {
            *(outCol.image) = *(inCol.image);
            *(outCol.variance) = *(inCol.variance);
            *(outCol.mask) = *(inCol.mask) | orMask;
        }
    }
}

//
// Explicit instantiations
//
//template void lsst::fw::kernel::printKernel<float>(
//    lsst::fw::Kernel<PixelT> const &kernel,
//    double x,
//    double y,
//    bool doNormalize,
//    std::string pixelFmt);
//template void lsst::fw::kernel::printKernel<double>(
//    lsst::fw::Kernel<PixelT> const &kernel,
//    double x,
//    double y,
//    bool doNormalize,
//    std::string pixelFmt);
