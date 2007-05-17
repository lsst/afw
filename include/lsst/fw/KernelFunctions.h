// -*- LSST-C++ -*-
#ifndef LLST_FW_KernelFunctions_H
#define LLST_FW_KernelFunctions_H
/**
 * \file
 * \ingroup fw
 *
 * Convolve and apply functions for kernels
 *
 * To do:
 * - Mark pixels whose data comes from extended pixels by setting a bit in the mask
 *   One question is whether to mark all such pixels or only those that have
 *   "significant" (kernel>threshold) contribution.
 *
 * - Add versions of these functions that work with lsst::fw::Image
 *   This is not a high priority (not needed for DC2).
 *   One could immplement it by calling VW routines or just do it "manually".
 *
 * - Try a version of convolve (perhaps simplified) that uses a PixelProcessingFunction;
 *   compare the speed of that with the speed using MaskePixelAccessor.
 *   To implement the processing function version:
 *   - Run the function over a subimage that excludes the right and lower edges
 *   - Within the function save a copy of the PixelLocator, use ++ to increment columns
 *     and advance the initial copy to increment rows
 *
 * \author Russell Owen
 */
#include <vw/Image.h>

#include <lsst/fw/Image.h>
#include <lsst/fw/MaskedImage.h>
#include <lsst/fw/MaskedPixelAccessors.h>
#include <lsst/fw/Kernel.h>

namespace lsst {
namespace fw {

    template <class ImageT, class MaskT, class KernelAccessT>
    inline void apply2d(
        MaskedPixelAccessor<ImageT, MaskT> &resIter,
        MaskedPixelAccessor<ImageT, MaskT> const &imIter,
        KernelAccessT const &kIter,
        unsigned cols,
        unsigned rows,
        ImageT threshold
    );
    
    template <class ImageT, class MaskT, class KernelT, class EdgeT>
    void convolve2d(
        MaskedImage<ImageT, KernelT> const &maskedImage,
        Kernel<KernelT> const &kernel,
        ImageT const threshold,
        EdgeT const &extension
    );
    
    #include <lsst/fw/Kernel/KernelFunctions.cc>

}   // namespace fw
}   // namespace lsst

#endif // !defined(LLST_FW_KernelFunctions_H)
