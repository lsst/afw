#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE table-fits
#include "boost/test/unit_test.hpp"

#include <iostream>
#include <iterator>
#include <algorithm>
#include <map>

#include "lsst/afw/table/fits.h"
#include "lsst/afw/table/Simple.h"

BOOST_AUTO_TEST_CASE(testFits) {
    using namespace lsst::afw::table;

    lsst::afw::table::Schema schema(false);
    Key<int> a_b_i = schema.addField<int>("a.b.i", "int");
    Key<Flag> a_b_i_valid = schema.addField<Flag>("a.b.i.valid", "is field a.b.i valid?");
    Key<float> a_c_f = schema.addField<float>("a.c.f", "float", "femtoseamonkeys");
    Key<double> e_g_d = schema.addField<double>("e.g.d", "double", "bargles^2");
    Key<Flag> e_g_d_flag1 = schema.addField<Flag>("e.g.d.flag1", "flag1 for e.g.d");
    Key<Flag> e_g_d_flag2 = schema.addField<Flag>("e.g.d.flag2", "flag2 for e.g.d");
    Key< Point<float> > a_b_p = schema.addField< Point<float> >("a.b.p", "point", "pixels");

    SimpleTable table(schema);
    {
        SimpleRecord r1 = table.addRecord();
        r1.set(a_b_i, 314);
        r1.set(a_b_i_valid, true);
        r1.set(a_c_f, 3.14f);
        r1.set(e_g_d, 3.14E12);
        r1.set(e_g_d_flag1, false);
        r1.set(e_g_d_flag2, true);
        r1.set(a_b_p, lsst::afw::geom::Point2D(1.2, 0.5));
        SimpleRecord r2 = table.addRecord();
        r2.set(a_b_i, 5123);
        r2.set(a_b_i_valid, true);
        r2.set(a_c_f, 44.8f);
        r2.set(e_g_d, 12.2E-3);
        r2.set(e_g_d_flag1, true);
        r2.set(e_g_d_flag2, false);
        r2.set(a_b_p, lsst::afw::geom::Point2D(-32.1, 63.2));

        BOOST_CHECK_EQUAL( r2.get(e_g_d_flag1), true );
        BOOST_CHECK_EQUAL( r2.get(e_g_d_flag2), false );
    }

    table.writeFits("!testTable.fits");

    lsst::afw::fits::Fits file = lsst::afw::fits::Fits::openFile("testTable.fits[1]", true);
    file.checkStatus();
    lsst::afw::table::fits::readFitsHeader(file, true);
    file.closeFile();
    file.checkStatus();
}
