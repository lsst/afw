// -*- lsst-c++ -*-

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

#include <iostream>

#include <memory>
#include "boost/format.hpp"

#include "lsst/afw/image.h"
#include "lsst/afw/math.h"

using namespace std;

int main() {
    typedef lsst::afw::math::Kernel::Pixel Pixel;

    double majorSigma = 2.5;
    double minorSigma = 2.0;
    double angle = 0.5;
    unsigned int kernelCols = 5;
    unsigned int kernelRows = 4;

    lsst::afw::math::GaussianFunction2<Pixel> gaussFunc(majorSigma, minorSigma, angle);
    lsst::afw::math::AnalyticKernel analyticKernel(kernelCols, kernelRows, gaussFunc);
    lsst::afw::image::Image<Pixel> analyticImage(analyticKernel.getDimensions());
    (void)analyticKernel.computeImage(analyticImage, true);
    analyticImage *= 47.3; // denormalize by some arbitrary factor

    lsst::afw::math::FixedKernel fixedKernel(analyticImage);

    cout << boost::format("Gaussian kernel with majorSigma=%.1f, minorSigma=%.1f\n") %
        majorSigma % minorSigma;

    lsst::afw::math::printKernel(fixedKernel, true);
}
