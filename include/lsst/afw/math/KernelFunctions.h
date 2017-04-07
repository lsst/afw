// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#ifndef LSST_AFW_MATH_KERNELFUNCTIONS_H
#define LSST_AFW_MATH_KERNELFUNCTIONS_H
/*
 * Utility functions for kernels
 */

#include "lsst/afw/math/Kernel.h"

#include "lsst/afw/math/ConvolveImage.h"

namespace lsst {
namespace afw {
namespace math {

    /**
     * Print the pixel values of a Kernel to std::cout
     *
     * Rows increase upward and columns to the right; thus the lower left pixel is (0,0).
     *
     * @param kernel the kernel
     * @param doNormalize if true, normalize kernel
     * @param x x at which to evaluate kernel
     * @param y y at which to evaluate kernel
     * @param pixelFmt format for pixel values
     */
    void printKernel(
        lsst::afw::math::Kernel const &kernel,
        bool doNormalize,
        double x = 0,
        double y = 0,
        std::string pixelFmt = "%7.3f"
    );

}}}   // lsst::afw::math

#endif // !defined(LSST_AFW_MATH_KERNELFUNCTIONS_H)
