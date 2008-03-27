// -*- C++-LSST -*-
#ifndef LSST_AFW_IMAGE_PIXELACCESSORS.H
#define LSST_AFW_IMAGE_PIXELACCESSORS.H
/**
 * \file
 *
 * \brief Accessors for MaskedImage pixels (and perhaps eventually Image pixels)
 *
 * \author Russell Owen
 *
 * \ingroup afw
 */
#include <vw/Image.h>

#include <lsst/afw/image/MaskedImage.h>

namespace lsst {
namespace afw {
namespace image {

    /**
     * \brief Accessor for MaskedImage pixels
     *
     * Pixel data is accessed via pointers named image, variance and mask, e.g.:
     *     oldImageValue = *maskedPixelAccessor.image;
     *     *maskedPixelAccessor.image = newImageValue;
     *
     * The accessor is a very thin layer on VisualWorkbench's MemoryStridingPixelAccessor.
     * Note that there is no default (no arguments) constructor because it makes no sense
     * and because vw::MemoryStridingPixelAccessor does not have one.
     *
     * \ingroup afw
     */
    template <typename ImageT, typename MaskT>
    class MaskedPixelAccessor {
    public:
        typedef ImageT imagePixelType;
        typedef MaskT maskPixelType;
        typedef typename vw::ImageView<ImageT>::pixel_accessor imageAccessorType;
        typedef typename vw::ImageView<MaskT>::pixel_accessor maskAccessorType;
        
        /**
         * \brief Construct from a MaskedImage
         */
        explicit MaskedPixelAccessor(
            MaskedImage<ImageT, MaskT> &maskedImage)
        :
            image(maskedImage.getImage()->origin()),
            variance(maskedImage.getVariance()->origin()),
            mask(maskedImage.getMask()->origin())
        {}
        
        /**
         * \brief Construct from a const MaskedImage
         */
        explicit MaskedPixelAccessor(
            MaskedImage<ImageT, MaskT> const &maskedImage)
        :
            image(maskedImage.getImage()->origin()),
            variance(maskedImage.getVariance()->origin()),
            mask(maskedImage.getMask()->origin())
        {}
        
        /**
         * \brief Construct from three VW::MemoryStridingPixelAccessors
         */
        explicit MaskedPixelAccessor(
            imageAccessorType &imageAccessor,
            imageAccessorType &varianceAccessor,
            maskAccessorType &maskAccessor)
        :
            image(imageAccessor),
            variance(varianceAccessor),
            mask(maskAccessor)
        {}

        /**
         * \brief Point to the next column
         */
        inline void nextCol() {
            image.next_col();
            variance.next_col();
            mask.next_col();
        }

        /**
         * \brief Point to the previous column
         */
        inline void prevCol() {
            image.prev_col();
            variance.prev_col();
            mask.prev_col();
        }

        /**
         * \brief Point to the next row
         */
        inline void nextRow() {
            image.next_row();
            variance.next_row();
            mask.next_row();
        }

        /**
         * \brief Point to the previous row
         */
        inline void prevRow() {
            image.prev_row();
            variance.prev_row();
            mask.prev_row();
        }

        /**
         * \brief Point to the next plane
         */
        inline void nextPlane() {
            image.next_plane();
            variance.next_plane();
            mask.next_plane();
        }

        /**
         * \brief Point to the previous plane
         */
        inline void prevPlane() {
            image.prev_plane();
            variance.prev_plane();
            mask.prev_plane();
        }

        /**
         * \brief Advance by the specified amount
         */
        inline void advance(
            ptrdiff_t colOffset,    ///< column offset
            ptrdiff_t rowOffset,     ///< row offset
            ptrdiff_t planeOffset=0) ///< plane offset
        {
            image.advance(colOffset, rowOffset, planeOffset);
            variance.advance(colOffset, rowOffset, planeOffset);
            mask.advance(colOffset, rowOffset, planeOffset);
        }

        imageAccessorType image;    ///< image pixel accessor
        imageAccessorType variance; ///< variance pixel accessor
        maskAccessorType mask;      ///< mask pixel accessor
    };

}}}  // lsst::afw::image

#endif // LSST_AFW_IMAGE_PIXELACCESSORS.H
