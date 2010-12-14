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
#include <cmath>
#include <vector>
#include <exception>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Kernel

#include "boost/make_shared.hpp"
#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/LocalKernel.h"


typedef lsst::afw::math::Kernel Kernel;
typedef lsst::afw::math::FixedKernel FixedKernel;
typedef lsst::afw::math::LinearCombinationKernel LinearCombinationKernel;
typedef lsst::afw::math::KernelList KernelList;
typedef lsst::afw::image::Image<Kernel::Pixel> Image;
typedef lsst::afw::math::LocalKernel LocalKernel;
typedef lsst::afw::math::ImageLocalKernel ImageLocalKernel;
typedef lsst::afw::math::FourierLocalKernel FourierLocalKernel;

BOOST_AUTO_TEST_CASE(LocalKernelTest) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    int width = 7, height = 7;
    Image img(width,height, 0);
    img(width/2 + 1, height/2 + 1) = 1;

    
    FixedKernel fixedKernel(img);

    FourierLocalKernel::Ptr fourierKernel;
    ImageLocalKernel::Ptr imgKernel;

    BOOST_CHECK_NO_THROW(imgKernel = fixedKernel.computeImageLocalKernel(
            lsst::afw::geom::makePointD(3.4, 0.8886))
    );
    BOOST_CHECK(imgKernel.get() != 0);
    BOOST_CHECK_NO_THROW(fourierKernel = fixedKernel.computeFourierLocalKernel(
            lsst::afw::geom::makePointD(0, 1))
    );
    BOOST_CHECK(fourierKernel.get() != 0);

    Image::Ptr imgFromLocalKernel = imgKernel->getImage();

    BOOST_CHECK_EQUAL(imgFromLocalKernel->getHeight(), height);
    BOOST_CHECK_EQUAL(imgFromLocalKernel->getWidth(), width);
    
    for(int y = 0; y < height; ++y) {
        Image::x_iterator vIter = imgFromLocalKernel->row_begin(y);
        Image::x_iterator vEnd = imgFromLocalKernel->row_end(y);
        Image::x_iterator iIter = img.row_begin(y);
        for(; vIter != vEnd; ++vIter, ++iIter) {
            BOOST_CHECK_CLOSE(static_cast<double>(*vIter), static_cast<double>(*iIter), 0.00001);
        }
    }
}
