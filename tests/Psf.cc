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
 
//  -*- lsst-c++ -*-
#include <iostream>
#include <string>
#include <algorithm>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Psf

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/math/Kernel.h"

/* This is a C++ version of testKernelPsf in Psf.py; makes it easier to run through valgrind. */

BOOST_AUTO_TEST_CASE(kernel_from_psf) {

    double x = 10.4999, y = 10.4999;
    int ksize = 15;
    int sigma1 = 1;

    lsst::afw::detection::Psf::Ptr kPsf = lsst::afw::detection::createPsf(
        "Kernel", boost::make_shared<lsst::afw::math::AnalyticKernel>(
            ksize, ksize, lsst::afw::math::GaussianFunction2<double>(sigma1, sigma1)
        )
    );
    lsst::afw::detection::Psf::Image::Ptr kIm = kPsf->computeImage(lsst::afw::geom::Point2D(x, y));

    lsst::afw::detection::Psf::Ptr dgPsf = lsst::afw::detection::createPsf(
        "DoubleGaussian", ksize, ksize, sigma1
    );
    lsst::afw::detection::Psf::Image::Ptr dgIm = dgPsf->computeImage(lsst::afw::geom::Point2D(x, y));

    lsst::afw::detection::Psf::Image diff(*kIm, true); diff -= *dgIm;
    lsst::afw::math::Statistics stats = lsst::afw::math::makeStatistics(
        diff, lsst::afw::math::MAX | lsst::afw::math::MIN
    );

    BOOST_CHECK_CLOSE( stats.getValue(lsst::afw::math::MAX), 0.0, 1E-16 );
    BOOST_CHECK_CLOSE( stats.getValue(lsst::afw::math::MIN), 0.0, 1E-16 );

}
