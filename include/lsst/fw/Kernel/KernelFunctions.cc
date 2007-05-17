// -*- LSST-C++ -*-
#include <vw/Image.h>

// #include <lsst/fw/KernelFunctions.h>

namespace lsst {
namespace fw {

    /**
     * \brief Apply convolution kernel to a masked image at one point,
     * computing resulting data for one pixel.
     *
     * Outputs
     * resIter: masked image accessor pointing to resulting pixel
     *
     * Inputs
     * imIter: masked image accessor pointing to image pixel
     *    that overlaps the upper left corner of the kernel(!)
     * kIter: kernel pixel accessor for upper left corner of the kernel
     * cols: number of columns to convolve (# of cols in the kernel)
     * rows: number of rows to convolve (# of rows in the kernel)
     *
     * Note: this is a high performance routine; the user is expected to:
     * - handle edge extension
     * - figure out the kernel center and adjust the supplied pixel accessors accordingly
     * For an example of how to do this see the convolve2d function.
     */
    template <class ImageT, class MaskT, class KernelAccessT>
    inline void apply2d(
        MaskedPixelAccessor<ImageT, MaskT> &resIter,
        MaskedPixelAccessor<ImageT, MaskT> const &imIter,
        KernelAccessT const &kIter,
        unsigned cols,
        unsigned rows,
        ImageT threshold
    ) {
        *resIter.camera = 0;
        *resIter.mask = 0;
        *resIter.variance = 0;

        MaskedPixelAccessor<ImageT, MaskT> imRow = imIter;
        KernelAccessT kRow = kIter;
        for (unsigned row = 0; row < rows; ++row) {
            for (unsigned col = 0; col < cols; ++col) {
                MaskedPixelAccessor<ImageT, MaskT> imCol = imRow;
                KernelAccessT kCol = kRow;

                *resIter.camera += (*kCol) * (*imRow.camera);
                if ((*imCol.mask) && ((*kCol) > threshold)) {
                    // this bad pixel contributes enough to "OR" in the bad bits
                    *resIter.mask |= *imCol.mask;
                }
                *resIter.variance += (*kCol) * (*kCol) * (*imCol.variance);
                
                imCol.nextCol();
                kCol.next_col();
            }
            imRow.nextRow();
            kRow.next_row();
        }
    }
    
    
    /**
     * Convolve a masked image with a kernel
     *
     * To do:
     * - Think harder about computing x and y for spatial variation
     */
    template <typename ImageT, typename MaskT, typename KernelT, typename EdgeT>
    void convolve2d(
        MaskedImage<ImageT, MaskT> const &maskedImage,
        Kernel<KernelT> const &kernel,
        ImageT const threshold,
        EdgeT const &extension
    ) {
        unsigned kCols = kernel.getCols();
        unsigned kRows = kernel.getRows();
        
        MaskedPixelAccessor<ImageT, MaskT> imRow;
        
        unsigned outCols = 0;
        unsigned outRows = 0;
        typename Image<ImageT>::pixel_accessor imCamIter;
        typename Image<MaskT>::pixel_accessor imMaskIter;
        typename Image<ImageT>::pixel_accessor imVarIter;

        if (!boost::is_same<EdgeT, vw::NoEdgeExtension>::value) {
            // edge-extend the image; output image is same size as input image
            outCols = maskedImage.getCols();
            outRows = maskedImage.getRows();

            int extColOffset = - kernel.getCtrCol();
            int extRowOffset = - kernel.getCtrRow();
            unsigned extCols = outCols + kernel.getCols();
            unsigned extRows = outRows + kernel.getRows();
            Image<ImageT>extImCam = vw::EdgeExtensionView<vw::ImageView<ImageT>, EdgeT>(
                maskedImage.getImage().getIVwPtr(), extColOffset, extRowOffset, extCols, extRows);
            Image<MaskT>extImMask = vw::EdgeExtensionView<vw::ImageView<MaskT>, EdgeT>(
                maskedImage.getMask().getIVwPtr(), extColOffset, extRowOffset, extCols, extRows);
            Image<ImageT>extImVar = vw::EdgeExtensionView<vw::ImageView<ImageT>, EdgeT>(
                maskedImage.getVariance().getIVwPtr(), extColOffset, extRowOffset, extCols, extRows);
            
            imRow = MaskedPixelAccessor<ImageT, MaskT>(extImCam.origin(), extImMask.origin(), extImVar.origin());
        } else {
            // no edge extension wanted; output image is smaller than input image
            outCols = maskedImage.getCols() - kernel.getCols();
            outRows = maskedImage.getRows() - kernel.getRows();
            if ((outCols < 1) || (outRows < 1)) {
                throw std::invalid_argument("Image must be larger than kernel");
            }
            
            imRow = MaskedPixelAccessor<ImageT, MaskT>(maskedImage);
        }

        MaskedImage<ImageT, MaskT> resMaskedImage(outCols, outRows);
        MaskedPixelAccessor<ImageT, MaskT> resRow(resMaskedImage);

        if (kernel.isSpatiallyVarying()) {
            for (unsigned row = 0; row < outRows; row++) {
                MaskedPixelAccessor<ImageT, MaskT> imCol = imRow;
                MaskedPixelAccessor<ImageT, MaskT> resCol = resRow;
                for (unsigned col = 0; col < outCols; col++) {
                    KernelT x = static_cast<ImageT>(col); // probably not right
                    KernelT y = static_cast<ImageT>(row);
                    typename Image<KernelT>::pixel_accessor kIter = (*kernel.getImage(x, y).getIVwPtr()).origin();
                    apply2d(resCol, imCol, kIter, kCols, kRows, threshold);
                    resCol.nextCol();
                    imCol.nextCol();
                }
                resRow.nextRow();
                imRow.nextRow();
            }
        } else {
            typename Image<KernelT>::pixel_accessor kIter = (*kernel.getImage().getIVwPtr()).origin();
            for (unsigned row = 0; row < outRows; row++) {
                MaskedPixelAccessor<ImageT, MaskT> imCol = imRow;
                MaskedPixelAccessor<ImageT, MaskT> resCol = resRow;
                for (unsigned col = 0; col < outCols; col++) {
                    apply2d(resCol, imCol, kIter, kCols, kRows, threshold);
                    resCol.nextCol();
                    imCol.nextCol();
                }
                resRow.nextRow();
                imRow.nextRow();
            }
        }
        return resMaskedImage;
    } 

}   // namespace fw
}   // namespace lsst
