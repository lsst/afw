// -*- C++-LSST -*-
#ifndef LLST_FW_PixelAccessors_H
#define LLST_FW_PixelAccessors_H
/**
 * \file
 * \ingroup fw
 *
 * Pixel Accessors for MaskedImage.
 *
 * To do:
 * - Test speed of convolution using PixelLocator; if adequate then ditch this
 * - If retained then separate implementation into a separate file
 *
 * \author Russell Owen
 */
#include <vw/Image.h>

#include <lsst/fw/MaskedImage.h>

namespace lsst {
namespace fw {

    /**
     * Accessor for MaskedImage pixels
     *
     * A think layer on VisualWorkbench's pixel accessor
     */
    template<typename ImageT, typename MaskT>
    class MaskedPixelAccessor
    {
    public:
        MaskedPixelAccessor(
            MaskedImage<ImageT, MaskT> &maskedImage
        ) :
            camera((maskedImage.getImage().getIVwPtr()).origin()),
            mask((maskedImage.getMask().getIVwPtr()).origin()),
            variance((maskedImage.getVariance().getIVwPtr()).origin())
        {}

        MaskedPixelAccessor(
            vw::MemoryStridingPixelAccessor<ImageT> cameraAccessor,
            vw::MemoryStridingPixelAccessor<MaskT> maskAccessor,
            vw::MemoryStridingPixelAccessor<ImageT> varianceAccessor
        ) :
            camera(cameraAccessor),
            mask(maskAccessor),
            variance(varianceAccessor)
        {}

        inline void nextCol() {
            camera.next_col();
            mask.next_col();
            variance.next_col();
        }

        inline void prevCol() {
            camera.prev_col();
            mask.prev_col();
            variance.prev_col();
        }

        inline void nextRow() {
            camera.next_row();
            mask.next_row();
            variance.next_row();
        }

        inline void prevRow() {
            camera.prev_row();
            mask.prev_row();
            variance.prev_row();
        }

        inline void nextPlane() {
            camera.next_plane();
            mask.next_plane();
            variance.next_plane();
        }

        inline void prevPlane() {
            camera.prev_plane();
            mask.prev_plane();
            variance.prev_plane();
        }

        inline void advance(ptrdiff_t dc, ptrdiff_t dr, ptrdiff_t dp=0) {
            camera.advance(dc, dr, dp);
            mask.advance(dc, dr, dp);
            variance.advance(dc, dr, dp);
        }

        vw::MemoryStridingPixelAccessor<ImageT> camera;
        vw::MemoryStridingPixelAccessor<MaskT> mask;
        vw::MemoryStridingPixelAccessor<ImageT> variance;
        
    };

}   // namespace lsst::fw
}   // namespace lsst

#endif // !defined(LLST_FW_PixelAccessors_H)
