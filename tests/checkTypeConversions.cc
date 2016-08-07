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
#include <string>
#include <algorithm>
#include <typeinfo>

#include "boost/mpl/for_each.hpp"
#include "boost/mpl/vector.hpp"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE MaskedImage

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/utils/Demangle.h"
#include "lsst/afw/image/lsstGil.h"

/************************************************************************************************************/

using namespace std;
namespace gil = boost::gil;
//
// Check the conversion from SrcImageT to DstImageT
//
template<typename SrcImageT>
struct do_check_conversion1 {
    template<typename DstImageT> void operator()(DstImageT&) {
        SrcImageT src_(10, 10);
        typename SrcImageT::view_t src(view(src_));
        src(0,0) = 100;

        DstImageT dst_(10, 10);
        typename DstImageT::view_t dst(view(dst_));

        copy_and_convert_pixels(src, dst);

        if (dst(0,0) != src(0,0)) {
            // The BOOST_CHECK message is uninformative, to print our own.
            cerr <<
                lsst::utils::demangleType(typeid(SrcImageT).name()) << " ---- " <<
                lsst::utils::demangleType(typeid(DstImageT).name()) << " " << dst(0,0) << " != " <<
                src(0,0) << " ";
            BOOST_CHECK(src(0,0) == dst(0,0)); // this will fail
        }
    }
};

template<typename TYPES>
struct do_check_conversion {
    template<typename U>
    void operator()(U&) {
        boost::mpl::for_each<TYPES>(do_check_conversion1<U>());
    }
};

BOOST_AUTO_TEST_CASE(convertGilTypes) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    // List of types; we'll check all N^2 conversions between them
    typedef boost::mpl::vector<
        gil::gray8_image_t,
        gil::gray16_image_t,
        gil::gray16s_image_t,
        gil::gray32s_image_t,
        gil::gray32f_noscale_image_t,
        gil::gray64f_noscale_image_t
    > image_types;

    boost::mpl::for_each<image_types>(do_check_conversion<image_types>());
}
