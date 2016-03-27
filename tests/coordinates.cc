// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ellipses
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/format.hpp"

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"

// Most Extent and Point operators are tested in coordinates.py in Python, but
// division of negative integers has different behavior in the two languages,
// so we test the C++ version here.
BOOST_AUTO_TEST_CASE(Operators) {
    lsst::afw::geom::Extent2I e1(12, -23);
    BOOST_CHECK_EQUAL(e1/4, lsst::afw::geom::Extent2I(e1.getX()/4, e1.getY()/4));
    lsst::afw::geom::Extent2I e2(e1);
    e2 /= 3;
    BOOST_CHECK_EQUAL(e2, lsst::afw::geom::Extent2I(e1.getX()/3, e1.getY()/3));
}
