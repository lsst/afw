// -*- LSST-C++ -*-
#ifndef LSST_AFW_MATH_CONVOLVEIMAGE_H
#define LSST_AFW_MATH_CONVOLVEIMAGE_H
/**
 * @file
 *
 * @brief Convolve and apply functions for Image and Kernel
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
#include "lsst/afw/math/Kernel.h"

namespace lsst {
namespace afw {
namespace math {

    template <typename OutImageT, typename InImageT>
    inline typename OutImageT::SinglePixel apply(
        typename InImageT::const_xy_locator& inLocator,
        typename lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::const_xy_locator& kernelLocator,
        int kWidth, int kHeight);
    
    template <typename OutImageT, typename InImageT>
    inline typename OutImageT::SinglePixel apply(
        typename InImageT::const_xy_locator& inImage,
        std::vector<lsst::afw::math::Kernel::PixelT> const& kernelColList,
        std::vector<lsst::afw::math::Kernel::PixelT> const& kernelRowList
    );
    
    template <typename OutImageT, typename InImageT>
    void basicConvolve(
        OutImageT& convolvedImage,
        InImageT const& inImage,
        lsst::afw::math::Kernel const& kernel,
        bool doNormalize
    );
    
    template <typename OutImageT, typename InImageT>
    void basicConvolve(
        OutImageT& convolvedImage,
        InImageT const& inImage,
        lsst::afw::math::DeltaFunctionKernel const& kernel,
        bool doNormalize
    );
    
    template <typename OutImageT, typename InImageT>
    void basicConvolve(
        OutImageT& convolvedImage,
        InImageT const& inImage,
        lsst::afw::math::SeparableKernel const& kernel,
        bool doNormalize
    );
    
    template <typename OutImageT, typename InImageT, typename KernelT>
    void convolve(
        OutImageT& convolvedImage,
        InImageT const& inImage,
        KernelT const& kernel,
        bool doNormalize,
        int edgeBit=-1
    );

    template <typename OutImageT, typename InImageT>
    void convolveLinear(
        OutImageT& convolvedImage,
        InImageT const& inImage,
        lsst::afw::math::LinearCombinationKernel const& kernel,
        int edgeBit=-1
    );
}}}   // lsst::afw::math

//
// lsst/afw/math/ConvolveImage.cc has moved to src/math and all needed convolutions
// are explicitly instantiated --- probably with full and aggressive optimisation
//

#endif // !defined(LSST_AFW_MATH_CONVOLVEIMAGE_H)
