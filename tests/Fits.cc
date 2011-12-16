#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE table-fits
#include "boost/test/unit_test.hpp"

#include <iostream>
#include <iterator>
#include <algorithm>
#include <map>

#include "lsst/afw/table/fits.h"

BOOST_AUTO_TEST_CASE(testFits) {
    lsst::afw::fits::Fits fits = lsst::afw::fits::Fits::createFile("!testTable.fits");
    fits.checkStatus();
    lsst::afw::table::Schema schema(true);
    schema.addField<int>("a.b.i", "int");
    schema.addField<lsst::afw::table::Flag>("a.b.i.valid", "is field a.b.i valid?");
    schema.addField<float>("a.c.f", "float", "femtoseamonkeys");
    schema.addField<double>("e.g.d", "double", "bargles^2");
    schema.addField<lsst::afw::table::Flag>("e.g.d.flag1", "flag1 for e.g.d");
    schema.addField<lsst::afw::table::Flag>("e.g.d.flag2", "flag2 for e.g.d");
    schema.addField< lsst::afw::table::Point<float> >("a.b.p", "point", "pixels");
    lsst::afw::table::createFitsHeader(fits, schema, true);
    fits.closeFile();
    fits.checkStatus();
}
