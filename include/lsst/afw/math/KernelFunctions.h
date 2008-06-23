// -*- LSST-C++ -*-
#ifndef LSST_AFW_MATH_KERNELFUNCTIONS_H
#define LSST_AFW_MATH_KERNELFUNCTIONS_H
/**
 * \file
 *
 * \brief Convolve and apply functions for kernels
 *
 * - Add versions of these functions that work with lsst::afw::image::Image
 *   This is not a high priority (not needed for DC2).
 *
 * \author Russell Owen
 *
 * \ingroup afw
 */
#include "vw/Image.h"
#include "vw/Math/BBox.h"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/image/PixelAccessors.h"
#include "lsst/afw/math/Kernel.h"

namespace lsst {
namespace afw {
namespace math {

    template <typename ImageT, typename MaskT>
    inline void apply(
        lsst::afw::image::MaskedPixelAccessor<ImageT, MaskT> &outAccessor,
        lsst::afw::image::MaskedPixelAccessor<ImageT, MaskT> const &imageAccessor,
        typename lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::pixel_accessor const &kernelAccessor,
        unsigned int cols,
        unsigned int rows
    );
    
    template <typename ImageT, typename MaskT, typename KernelT>
    void basicConvolve(
        lsst::afw::image::MaskedImage<ImageT, MaskT> &convolvedImage,
        lsst::afw::image::MaskedImage<ImageT, MaskT> const &maskedImage,
        KernelT const &kernel,
        bool doNormalize
    );
    
    template <typename ImageT, typename MaskT>
    void basicConvolve(
        lsst::afw::image::MaskedImage<ImageT, MaskT> &convolvedImage,
        lsst::afw::image::MaskedImage<ImageT, MaskT> const &maskedImage,
        lsst::afw::math::DeltaFunctionKernel const &kernel,
        bool doNormalize
    );
    
    template <typename ImageT, typename MaskT, typename KernelT>
    void convolve(
        lsst::afw::image::MaskedImage<ImageT, MaskT> &convolvedImage,
        lsst::afw::image::MaskedImage<ImageT, MaskT> const &maskedImage,
        KernelT const &kernel,
        int edgeBit,
        bool doNormalize
    );
    
    template <typename ImageT, typename MaskT, typename KernelT>
    lsst::afw::image::MaskedImage<ImageT, MaskT> convolve(
        lsst::afw::image::MaskedImage<ImageT, MaskT> const &maskedImage,
        KernelT const &kernel,
        int edgeBit,
        bool doNormalize
    );

    template <typename ImageT, typename MaskT>
    void convolveLinear(
        lsst::afw::image::MaskedImage<ImageT, MaskT> &convolvedImage,
        lsst::afw::image::MaskedImage<ImageT, MaskT> const &maskedImage,
        lsst::afw::math::LinearCombinationKernel const &kernel,
        int edgeBit
    );

    template <typename ImageT, typename MaskT>
    lsst::afw::image::MaskedImage<ImageT, MaskT> convolveLinear(
        lsst::afw::image::MaskedImage<ImageT, MaskT> const &maskedImage,
        lsst::afw::math::LinearCombinationKernel const &kernel,
        int edgeBit
    );

    void printKernel(
        lsst::afw::math::Kernel const &kernel,
        bool doNormalize,
        double x = 0,
        double y = 0,
        std::string pixelFmt = "%7.3f"
    );

}}}   // lsst::afw::math
    
#ifndef SWIG // don't bother SWIG with .cc files
#include "lsst/afw/math/KernelFunctions.cc"
#endif

#endif // !defined(LSST_AFW_MATH_KERNELFUNCTIONS_H)
