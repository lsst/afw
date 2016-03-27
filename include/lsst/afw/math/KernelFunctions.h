// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#ifndef LSST_AFW_MATH_KERNELFUNCTIONS_H
#define LSST_AFW_MATH_KERNELFUNCTIONS_H
/**
 * @file
 *
 * @brief Utility functions for kernels
 *
 * @todo
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */

#include "lsst/afw/math/Kernel.h"

#include "lsst/afw/math/ConvolveImage.h"

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
