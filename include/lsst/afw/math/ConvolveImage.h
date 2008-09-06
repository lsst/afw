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
#include "vw/Image.h"
#include "vw/Math/BBox.h"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Kernel.h"

namespace lsst {
namespace afw {
namespace math {

    template <typename OutPixelT, typename InPixelT>
    inline void apply(
        OutPixelT &outPixel,
        typename lsst::afw::image::Image<InPixelT>::pixel_accessor const &imageAccessor,
        typename lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::pixel_accessor const &kernelAccessor,
        unsigned int cols,
        unsigned int rows
    );
    
    template <typename OutPixelT, typename InPixelT>
    inline void apply(
        OutPixelT &outPixel,
        typename lsst::afw::image::Image<InPixelT>::pixel_accessor const &imageAccessor,
        std::vector<lsst::afw::math::Kernel::PixelT> const &kernelColList,
        std::vector<lsst::afw::math::Kernel::PixelT> const &kernelRowList
    );
    
    template <typename OutPixelT, typename InPixelT>
    void basicConvolve(
        lsst::afw::image::Image<OutPixelT> &convolvedImage,
        lsst::afw::image::Image<InPixelT> const &inImage,
        lsst::afw::math::Kernel const &kernel,
        bool doNormalize
    );
    
    template <typename OutPixelT, typename InPixelT>
    void basicConvolve(
        lsst::afw::image::Image<OutPixelT> &convolvedImage,
        lsst::afw::image::Image<InPixelT> const &inImage,
        lsst::afw::math::DeltaFunctionKernel const &kernel,
        bool doNormalize
    );
    
    template <typename OutPixelT, typename InPixelT>
    void basicConvolve(
        lsst::afw::image::Image<OutPixelT> &convolvedImage,
        lsst::afw::image::Image<InPixelT> const &inImage,
        lsst::afw::math::SeparableKernel const &kernel,
        bool doNormalize
    );
    
    template <typename OutPixelT, typename InPixelT, typename KernelT>
    void convolve(
        lsst::afw::image::Image<OutPixelT> &convolvedImage,
        lsst::afw::image::Image<InPixelT> const &inImage,
        KernelT const &kernel,
        bool doNormalize
    );
    
    template <typename InPixelT, typename KernelT>
    lsst::afw::image::Image<InPixelT> convolveNew(
        lsst::afw::image::Image<InPixelT> const &inImage,
        KernelT const &kernel,
        bool doNormalize
    );

    template <typename OutPixelT, typename InPixelT>
    void convolveLinear(
        lsst::afw::image::Image<OutPixelT> &convolvedImage,
        lsst::afw::image::Image<InPixelT> const &inImage,
        lsst::afw::math::LinearCombinationKernel const &kernel
    );

    template <typename InPixelT>
    lsst::afw::image::Image<InPixelT> convolveLinearNew(
        lsst::afw::image::Image<InPixelT> const &inImage,
        lsst::afw::math::LinearCombinationKernel const &kernel
    );
}}}   // lsst::afw::math
    
#ifndef SWIG // don't bother SWIG with .cc files
#include "lsst/afw/math/ConvolveImage.cc"
#endif

#endif // !defined(LSST_AFW_MATH_CONVOLVEIMAGE_H)
