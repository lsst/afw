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

/*
 * Definitions of members of lsst::afw::math
 */
#include <iostream>

#include "boost/format.hpp"

#include "lsst/afw/math/KernelFunctions.h"

namespace lsst {
namespace afw {
namespace math {

void printKernel(Kernel const &kernel, bool doNormalize, double xPos, double yPos, std::string pixelFmt) {
    typedef Kernel::Pixel Pixel;

    image::Image<Pixel> kImage(kernel.getDimensions());
    double kSum = kernel.computeImage(kImage, doNormalize, xPos, yPos);

    for (int y = kImage.getHeight() - 1; y >= 0; --y) {
        for (image::Image<Pixel>::const_x_iterator ptr = kImage.row_begin(y); ptr != kImage.row_end(y);
             ++ptr) {
            std::cout << boost::format(pixelFmt) % *ptr << " ";
        }
        std::cout << std::endl;
    }

    if (doNormalize && std::abs(static_cast<double>(kSum) - 1.0) > 1.0e-5) {
        std::cout << boost::format("Warning! Sum of all pixels = %9.5f != 1.0\n") % kSum;
    }
    std::cout << std::endl;
}
}
}
}  // end lsst::afw::math
