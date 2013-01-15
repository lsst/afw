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
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Peak

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/detection/Peak.h"

namespace lsst { namespace afw { namespace detection {

BOOST_AUTO_TEST_CASE(CCtorAndAssignment) {
    pex::logging::Trace::setVerbosity("afw.detection", 0);

    Peak peak1(10, 20);
    Peak peak2;
    peak2 = peak1;                      // test assignment

    BOOST_CHECK(peak2.getId() != peak1.getId());
    BOOST_CHECK(peak2.getCentroid() == peak1.getCentroid());

    Peak peak3(peak1);                  // test cctor
    BOOST_CHECK(peak3.getId() != peak1.getId());
    BOOST_CHECK(peak3.getCentroid() == peak1.getCentroid());
}
}}}
