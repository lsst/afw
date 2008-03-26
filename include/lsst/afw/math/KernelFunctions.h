// -*- LSST-C++ -*-
#ifndef LSST_AFW_MATH_KERNELFUNCTIONS_H
#define LSST_AFW_MATH_KERNELFUNCTIONS_H
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

#include <lsst/afw/image/Image.h>
#include <lsst/afw/image/MaskedImage.h>
#include <lsst/afw/image/PixelAccessors.h>
#include <lsst/afw/math/Kernel.h>

namespace lsst {
namespace fw {
namespace kernel {

    template <typename ImageT, typename MaskT>
    inline void apply(
        lsst::fw::MaskedPixelAccessor<ImageT, MaskT> &outAccessor,
        lsst::fw::MaskedPixelAccessor<ImageT, MaskT> const &imageAccessor,
        typename lsst::fw::Image<lsst::fw::Kernel::PixelT>::pixel_accessor const &kernelAccessor,
        unsigned int cols,
        unsigned int rows
    );
    
    template <typename ImageT, typename MaskT, typename KernelT>
    void basicConvolve(
        lsst::fw::MaskedImage<ImageT, MaskT> &convolvedImage,
        lsst::fw::MaskedImage<ImageT, MaskT> const &maskedImage,
        KernelT const &kernel,
        bool doNormalize
    );
    
    template <typename ImageT, typename MaskT>
    void basicConvolve(
        lsst::fw::MaskedImage<ImageT, MaskT> &convolvedImage,
        lsst::fw::MaskedImage<ImageT, MaskT> const &maskedImage,
        lsst::fw::DeltaFunctionKernel const &kernel,
        bool doNormalize
    );
    
    template <typename ImageT, typename MaskT, typename KernelT>
    void convolve(
        lsst::fw::MaskedImage<ImageT, MaskT> &convolvedImage,
        lsst::fw::MaskedImage<ImageT, MaskT> const &maskedImage,
        KernelT const &kernel,
        int edgeBit,
        bool doNormalize
    );
    
    template <typename ImageT, typename MaskT, typename KernelT>
    lsst::fw::MaskedImage<ImageT, MaskT> convolve(
        lsst::fw::MaskedImage<ImageT, MaskT> const &maskedImage,
        KernelT const &kernel,
        int edgeBit,
        bool doNormalize
    );

    template <typename ImageT, typename MaskT>
    void convolveLinear(
        lsst::fw::MaskedImage<ImageT, MaskT> &convolvedImage,
        lsst::fw::MaskedImage<ImageT, MaskT> const &maskedImage,
        lsst::fw::LinearCombinationKernel const &kernel,
        int edgeBit
    );

    template <typename ImageT, typename MaskT>
    lsst::fw::MaskedImage<ImageT, MaskT> convolveLinear(
        lsst::fw::MaskedImage<ImageT, MaskT> const &maskedImage,
        lsst::fw::LinearCombinationKernel const &kernel,
        int edgeBit
    );

    void printKernel(
        lsst::fw::Kernel const &kernel,
        double x = 0,
        double y = 0,
        bool doNormalize = true,
        std::string pixelFmt = "%7.3f"
    );

}}}   // lsst::fw::kernel
    
#ifndef SWIG // don't bother SWIG with .cc files
#include <lsst/afw/math/KernelFunctions.cc>
#endif

#endif // !defined(LSST_AFW_MATH_KERNELFUNCTIONS_H)
