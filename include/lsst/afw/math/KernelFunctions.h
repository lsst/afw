// -*- LSST-C++ -*-
#ifndef LSST_AFW_MATH_KERNELFUNCTIONS_H
#define LSST_AFW_MATH_KERNELFUNCTIONS_H
/**
 * @file
 *
 * @brief Convolve and apply functions for kernels
 *
 * @todo
 * * Add versions of these functions that work with lsst::afw::image::Image.
 *   This is not a high priority because it is not needed for DC3.
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

#include "lsst/afw/math/Kernel.h"

#include "lsst/afw/math/ConvolveImage.h"
#include "lsst/afw/math/ConvolveMaskedImage.h"

namespace lsst {
namespace afw {
namespace math {

    void printKernel(
        lsst::afw::math::Kernel const &kernel,
        bool doNormalize,
        double x = 0,
        double y = 0,
        std::string pixelFmt = "%7.3f"
    );

}}}   // lsst::afw::math

#endif // !defined(LSST_AFW_MATH_KERNELFUNCTIONS_H)
