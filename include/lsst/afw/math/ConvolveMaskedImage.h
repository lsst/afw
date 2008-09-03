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
#include "vw/Image.h"
#include "vw/Math/BBox.h"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/image/PixelAccessors.h"
#include "lsst/afw/math/Kernel.h"

namespace lsst {
namespace afw {
namespace math {

    template <typename OutPixelT, typename InPixelT, typename MaskPixelT>
    inline void apply(
        lsst::afw::image::MaskedPixelAccessor<OutPixelT, MaskPixelT> &outAccessor,
        lsst::afw::image::MaskedPixelAccessor<InPixelT, MaskPixelT> const &imageAccessor,
        typename lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::pixel_accessor const &kernelAccessor,
        unsigned int cols,
        unsigned int rows
    );
    
    template <typename OutPixelT, typename InPixelT, typename MaskPixelT>
    inline void apply(
        lsst::afw::image::MaskedPixelAccessor<InPixelT, MaskPixelT> &outAccessor,
        lsst::afw::image::MaskedPixelAccessor<OutPixelT, MaskPixelT> const &imageAccessor,
        std::vector<lsst::afw::math::Kernel::PixelT> const &kernelColList,
        std::vector<lsst::afw::math::Kernel::PixelT> const &kernelRowList
    );
    
    template <typename OutPixelT, typename InPixelT, typename MaskPixelT>
    void basicConvolve(
        lsst::afw::image::MaskedImage<OutPixelT, MaskPixelT> &convolvedImage,
        lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> const &maskedImage,
        lsst::afw::math::Kernel const &kernel,
        bool doNormalize
    );
    
    template <typename OutPixelT, typename InPixelT, typename MaskPixelT>
    void basicConvolve(
        lsst::afw::image::MaskedImage<OutPixelT, MaskPixelT> &convolvedImage,
        lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> const &maskedImage,
        lsst::afw::math::DeltaFunctionKernel const &kernel,
        bool doNormalize
    );
    
    template <typename OutPixelT, typename InPixelT, typename MaskPixelT>
    void basicConvolve(
        lsst::afw::image::MaskedImage<OutPixelT, MaskPixelT> &convolvedImage,
        lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> const &maskedImage,
        lsst::afw::math::SeparableKernel const &kernel,
        bool doNormalize
    );
    
    template <typename OutPixelT, typename InPixelT, typename MaskPixelT, typename KernelT>
    void convolve(
        lsst::afw::image::MaskedImage<OutPixelT, MaskPixelT> &convolvedImage,
        lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> const &maskedImage,
        KernelT const &kernel,
        int edgeBit,
        bool doNormalize
    );
    
    template <typename InPixelT, typename MaskPixelT, typename KernelT>
    lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> convolve(
        lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> const &maskedImage,
        KernelT const &kernel,
        int edgeBit,
        bool doNormalize
    );

    template <typename OutPixelT, typename InPixelT, typename MaskPixelT>
    void convolveLinear(
        lsst::afw::image::MaskedImage<OutPixelT, MaskPixelT> &convolvedImage,
        lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> const &maskedImage,
        lsst::afw::math::LinearCombinationKernel const &kernel,
        int edgeBit
    );

    template <typename InPixelT, typename MaskPixelT>
    lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> convolveLinear(
        lsst::afw::image::MaskedImage<InPixelT, MaskPixelT> const &maskedImage,
        lsst::afw::math::LinearCombinationKernel const &kernel,
        int edgeBit
    );

}}}   // lsst::afw::math
    
#ifndef SWIG // don't bother SWIG with .cc files
#include "lsst/afw/math/ConvolveMaskedImage.cc"
#endif

#endif // !defined(LSST_AFW_MATH_CONVOLVEMASKEDIMAGE_H)
