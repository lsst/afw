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

/*
 * An example executible which calls the example sex2dec code
 */
#include <iostream>
#include <string>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Coord

#include "boost/test/unit_test.hpp"

#include "lsst/geom.h"
#include "lsst/afw/coord.h"

#define CHECK_DIFF(x1, x2, d) BOOST_CHECK_SMALL(x1 - x2, d)

using namespace std;
namespace afwCoord = lsst::afw::coord;

BOOST_AUTO_TEST_CASE(dmsToDecimal) {
    std::string rastr = "10:00:00.00";
    std::string decstr = "-02:30:00.00";
    lsst::geom::Angle ra = afwCoord::hmsStringToAngle(rastr);
    lsst::geom::Angle dec = afwCoord::dmsStringToAngle(decstr);

    CHECK_DIFF(ra.asDegrees(), 150.0, 1e-8);
    CHECK_DIFF(dec.asDegrees(), -2.5, 1e-8);

    // make sure the rounding issue works (ie. 59.998 rounds to 00, not 60 sec)
    ra -= (0.000001 * lsst::geom::degrees);
    std::string raStr2 = afwCoord::angleToHmsString(ra);
    BOOST_CHECK_EQUAL(raStr2, rastr);
}
