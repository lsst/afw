

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

#ifndef LSST_AFW_MATH_CONVOLVEIMAGEGPU_H
#define LSST_AFW_MATH_CONVOLVEIMAGEGPU_H
/**
 * @file
 *
 * @brief Convolve functions for Image and Kernel, using GPU acceleration
 *
 * @author Kresimir Cosic (modifications for GPU)
 * @author Russell Owen (original code without GPU acceleration)
 *
 * @ingroup afw
 */
#include <limits>
#include <sstream>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"

namespace lsst {
namespace afw {
namespace math {

    template <typename OutImageT, typename InImageT, typename KernelT>
    void convolveGPU(
            OutImageT& convolvedImage,
            InImageT const& inImage,
            KernelT const& kernel,
            ConvolutionControl const& convolutionControl = ConvolutionControl());

    template <typename OutImageT, typename InImageT, typename KernelT>
    void convolveGPU(
            OutImageT& convolvedImage,
            InImageT const& inImage,
            KernelT const& kernel,
            bool doNormalize,
            bool doCopyEdge = false);


}}}   // lsst::afw::math

#endif // !defined(LSST_AFW_MATH_CONVOLVEIMAGE_H)

