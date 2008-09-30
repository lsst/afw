// -*- LSST-C++ -*-
#ifndef LSST_AFW_MATH_CONVOLVEMASKEDIMAGE_H
#define LSST_AFW_MATH_CONVOLVEMASKEDIMAGE_H
/**
 * @file
 *
 * @brief Convolve and apply functions for MaskedImage and Kernel
 *
 * @todo
 * * Consider adding a flag to convolve indicating which specialized version of basicConvolve was used.
 *   This would only be used for unit testing and trace messages suffice (barely), so not a high priority.
 * * Consider a way to disable use of specialized versions of basicConvolve.
 *   This could be used to replace convolveLinear with an automatic specialization.
 *   It might also be useful for unit tests to verify that the specialized version gives the same answer.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"

namespace lsst {
namespace afw {
namespace math {

    template <typename OutMaskedImageT, typename InMaskedImageT>
    inline void apply(
#if 0
        typename OutMaskedImageT::xy_locator& outLocator,
#else
        typename OutMaskedImageT::IMV_tuple const& outLocator, // triple of (image, mask, variance)
#endif
        typename InMaskedImageT::const_xy_locator& inLocator,
        lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::const_xy_locator& kernelLocator,
        int width, int height);
    
    template <typename OutMaskedImageT, typename InMaskedImageT>
    void apply(
        typename OutMaskedImageT::xy_locator& outLocator,
        typename InMaskedImageT::const_xy_locator& inLocator,
        std::vector<lsst::afw::math::Kernel::PixelT> const& kernelColList,
        std::vector<lsst::afw::math::Kernel::PixelT> const& kernelRowList);
    
    template <typename OutMaskedImageT, typename InMaskedImageT>
    void basicConvolve(
        OutMaskedImageT& convolvedImage,
        InMaskedImageT const& maskedImage,
        lsst::afw::math::Kernel const& kernel,
        bool doNormalize
    );
    
    template <typename OutMaskedImageT, typename InMaskedImageT>
    void basicConvolve(
        OutMaskedImageT& convolvedImage,
        InMaskedImageT const& maskedImage,
        lsst::afw::math::DeltaFunctionKernel const& kernel,
        bool doNormalize
    );
    
    template <typename OutMaskedImageT, typename InMaskedImageT>
    void basicConvolve(
        OutMaskedImageT& convolvedImage,
        InMaskedImageT const& maskedImage,
        lsst::afw::math::SeparableKernel const& kernel,
        bool doNormalize
    );
    
    template <typename OutMaskedImageT, typename InMaskedImageT, typename KernelT>
    void convolve(
        OutMaskedImageT& convolvedImage,
        InMaskedImageT const& maskedImage,
        KernelT const& kernel,
        int edgeBit,
        bool doNormalize
    );
    
    template <typename MaskedImageT, typename KernelT>
    MaskedImageT convolveNew(
        MaskedImageT const& maskedImage,
        KernelT const& kernel,
        int edgeBit,
        bool doNormalize
    );

    template <typename OutMaskedImageT, typename InMaskedImageT>
    void convolveLinear(
        OutMaskedImageT& convolvedImage,
        InMaskedImageT const& maskedImage,
        lsst::afw::math::LinearCombinationKernel const& kernel,
        int edgeBit
    );

    template <typename MaskedImageT>
    MaskedImageT convolveLinearNew(
        MaskedImageT const& maskedImage,
        lsst::afw::math::LinearCombinationKernel const& kernel,
        int edgeBit
    );

}}}   // lsst::afw::math
    
#ifndef SWIG // don't bother SWIG with .cc files
#include "lsst/afw/math/ConvolveMaskedImage.cc"
#endif

#endif // !defined(LSST_AFW_MATH_CONVOLVEMASKEDIMAGE_H)
