// -*- C++-LSST -*-
#ifndef LLST_FW_PixelAccessors_H
#define LLST_FW_PixelAccessors_H
/**
 * Pixel Accessors for MaskedImage.
 */
#include <vw/Image.h>

//#include "LsstBase.h"
//#include "MaskedImage.h"

namespace lsst {
namespace fw {

    /**
     * Accessor for MaskedImage pixels
     * modelled on VisualWorkbench's pixel accessor
     */
    template <class ImageT, class MaskT>
    class MaskedPixelAccessor // : lsst::fw::LsstBase
    {
    public:
        vw::MemoryStridingPixelAccessor<ImageT> camera;
        vw::MemoryStridingPixelAccessor<MaskT> mask;
        vw::MemoryStridingPixelAccessor<ImageT> variance;
        
//        MaskedPixelAccessor(lsst::fw::MaskedImage maskedImage)
//        :
//            camera((*maskedImage.getImage().getIVwPtr()).origin()),
//            mask((*maskedImage.getMask().getIVwPtr()).origin()),
//            variance((*maskedImage.getVariance().getIVwPtr()).origin())
//        {}

        MaskedPixelAccessor(
            vw::MemoryStridingPixelAccessor<ImageT> cameraAccessor,
            vw::MemoryStridingPixelAccessor<MaskT> maskAccessor,
            vw::MemoryStridingPixelAccessor<ImageT> varianceAccessor
        )
        :
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

        inline void advance() {
            camera.advance();
            mask.advance();
            variance.advance();
        }
    };

}   // namespace lsst::fw
}   // namespace lsst

#endif // !defined(LLST_FW_PixelAccessors_H)
