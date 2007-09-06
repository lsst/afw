// -*- LSST-C++ -*-
#ifndef LSST_FW_KernelFunctions_H
#define LSST_FW_KernelFunctions_H
/**
 * \file
 *
 * \brief Convolve and apply functions for kernels
 *
 * - Add versions of these functions that work with lsst::fw::Image
 *   This is not a high priority (not needed for DC2).
 *
 * \author Russell Owen
 *
 * \ingroup fw
 */
#include <vw/Image.h>
#include <vw/Math/BBox.h>

#include <lsst/fw/Image.h>
#include <lsst/fw/MaskedImage.h>
#include <lsst/fw/PixelAccessors.h>
#include <lsst/fw/Kernel.h>

namespace lsst {
namespace fw {
namespace kernel {

    template <typename ImageT, typename MaskT, typename KernelT>
    inline void apply(
        lsst::fw::MaskedPixelAccessor<ImageT, MaskT> &outAccessor,
        lsst::fw::MaskedPixelAccessor<ImageT, MaskT> const &imageAccessor,
        typename lsst::fw::Image<KernelT>::pixel_accessor const &kernelAccessor,
        unsigned int cols,
        unsigned int rows,
        KernelT threshold
    );
    
    template <typename ImageT, typename MaskT, typename KernelT>
    void convolve(
        lsst::fw::MaskedImage<ImageT, MaskT> &convolvedImage,
        lsst::fw::MaskedImage<ImageT, MaskT> const &maskedImage,
        lsst::fw::Kernel<KernelT> const &kernel,
        KernelT threshold,
        int edgeBit
    );
    
    template <typename ImageT, typename MaskT, typename KernelT>
    lsst::fw::MaskedImage<ImageT, MaskT> convolve(
        lsst::fw::MaskedImage<ImageT, MaskT> const &maskedImage,
        lsst::fw::Kernel<KernelT> const &kernel,
        KernelT threshold,
        int edgeBit
    );

    template <typename PixelT>
    void printKernel(
        lsst::fw::Kernel<PixelT> const &kernel,
        double x = 0,
        double y = 0,
        bool doNormalize = true,
        string pixelFmt = "%7.3f"
    );

    template <typename ImageT, typename MaskT>
    inline void _copyRegion(
        typename lsst::fw::MaskedImage<ImageT, MaskT> &destImage,
        typename lsst::fw::MaskedImage<ImageT, MaskT> const &sourceImage,
        vw::BBox2i const &region,
        MaskT orMask
    );

}}}   // lsst::fw::kernel
    
#ifndef SWIG // don't bother SWIG with .cc files
#include <lsst/fw/Kernel/KernelFunctions.cc>
#endif

#endif // !defined(LSST_FW_KernelFunctions_H)
