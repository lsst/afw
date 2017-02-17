#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE table-fits
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "lsst/afw/table/aggregates.h"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop

#include <iostream>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <map>

#include "boost/filesystem.hpp"

#include "lsst/afw/table/Source.h"
#include "lsst/afw/geom/Span.h"
#include "lsst/afw/geom/SpanSet.h"

struct EqualityCompare {

    bool operator()(
        lsst::afw::geom::Span const & a,
        lsst::afw::geom::Span const & b
    ) const {
        return a.getY() == b.getY() && a.getX0() == b.getX0() && a.getX1() == b.getX1();
    }

    bool operator()(float a, float b) const {
        return std::fabs(a - b) < 1E-8 || (std::isnan(a) && std::isnan(b));
    }

    bool operator()(
        lsst::afw::detection::PeakRecord const & a,
        lsst::afw::detection::PeakRecord const & b
    ) const {
        return (*this)(a.getFx(), b.getFx())
            && (*this)(a.getFy(), b.getFy())
            && (*this)(a.getPeakValue(), b.getPeakValue());
    }

};

struct ExtractSchemaStrings {

    template <typename T>
    void operator()(lsst::afw::table::SchemaItem<T> const & item) const {
        names.push_back(item.field.getName());
        docs.push_back(item.field.getDoc());
        units.push_back(item.field.getUnits());
    }

    mutable std::vector<std::string> names;
    mutable std::vector<std::string> docs;
    mutable std::vector<std::string> units;
};


BOOST_AUTO_TEST_CASE(testFits) {
    using namespace lsst::afw::table;

    std::string filename = "tests/data/testTable.fits";

    Schema schema = SourceTable::makeMinimalSchema();
    Key<int> a_b_i = schema.addField<int>("a_b_i", "int");
    Key<Flag> a_b_i_valid = schema.addField<Flag>("a_b_i_valid", "is field a.b.i valid?");
    Key<float> a_c_f = schema.addField<float>("a_c_f", "an extremely long string for documenting this float field that will require use of the FITS long-string convention that splits long values up and puts them on different keys using CONTINUE.", "barn");
    Key<double> e_g_d = schema.addField<double>("e_g_d", "double", "barn^2");
    Key<Flag> e_g_d_flag1 = schema.addField<Flag>("e_g_d_flag1", "flag1 for e.g.d");
    Key<Flag> e_g_d_flag2 = schema.addField<Flag>("e_g_d_flag2", "flag2 for e.g.d");
    PointKey<double> a_b_p = PointKey<double>::addFields(schema, "a_b_p", "point", "pixel");
    Key< std::string > a_s = schema.addField< std::string >("a_s", "string", 5);

    SourceCatalog vector(SourceTable::make(schema));

    vector.getTable()->setMetadata(std::make_shared<lsst::daf::base::PropertyList>());
    vector.getTable()->getMetadata()->add("SHEEP", 7.3, "total number of sheep on the farm");
    vector.getTable()->getMetadata()->add("MONKEYS", 155, "monkeys per tree");

    {
        PTR(Footprint) fp1 = std::make_shared<Footprint>();
        std::vector<lsst::afw::geom::Span> tempSpanList1 = {lsst::afw::geom::Span(0, 5, 8),
                                                            lsst::afw::geom::Span(1, 4, 9),
                                                            lsst::afw::geom::Span(2, 6, 7)};
        auto tempSpanSet1 = std::make_shared<lsst::afw::geom::SpanSet>(std::move(tempSpanList1));
        fp1->setSpans(tempSpanSet1);
        fp1->addPeak(4.5f, 1.2f, 25.6f);
        fp1->addPeak(6.8f, 0.8f, 23.2f);
        PTR(SourceRecord) r1 = vector.getTable()->makeRecord();
        r1->setFootprint(fp1);

        r1->set(a_b_i, 314);
        r1->set(a_b_i_valid, true);
        r1->set(a_c_f, 3.14f);
        r1->set(e_g_d, 3.14E12);
        r1->set(e_g_d_flag1, false);
        r1->set(e_g_d_flag2, true);
        r1->set(a_b_p, lsst::afw::geom::Point2D(1.2, 0.5));
        r1->set(a_s, "foo");
        vector.push_back(r1);

        PTR(SourceRecord) r2 = vector.getTable()->makeRecord();
        r2->set(a_b_i, 5123);
        r2->set(a_b_i_valid, true);
        r2->set(a_c_f, 44.8f);
        r2->set(e_g_d, 12.2E-3);
        r2->set(e_g_d_flag1, true);
        r2->set(e_g_d_flag2, false);
        r2->set(a_b_p, lsst::afw::geom::Point2D(-32.1, 63.2));
        r2->set(a_s, "bar");
        PTR(Footprint) fp2 = std::make_shared<Footprint>();
        std::vector<lsst::afw::geom::Span> tempSpanList2 = {lsst::afw::geom::Span(3, 2, 7),
                                                            lsst::afw::geom::Span(4, 3, 5)};
        auto tempSpanSet2 = std::make_shared<lsst::afw::geom::SpanSet>(std::move(tempSpanList2));
        fp2->setSpans(tempSpanSet2);
        fp2->addPeak(4.2f, 3.3f, 32.1f);
        r2->setFootprint(fp2);
        vector.push_back(r2);

        BOOST_CHECK_EQUAL( r2->get(e_g_d_flag1), true );
        BOOST_CHECK_EQUAL( r2->get(e_g_d_flag2), false );
    }

    vector.writeFits(filename);

    SourceCatalog readVector = SourceCatalog::readFits(filename);

    BOOST_CHECK_EQUAL( schema, readVector.getSchema() ); // only checks equality of keys

    BOOST_CHECK_EQUAL( readVector.getTable()->getMetadata()->get<double>("SHEEP"),
                       vector.getTable()->getMetadata()->get<double>("SHEEP") );
    BOOST_CHECK_EQUAL( readVector.getTable()->getMetadata()->get<int>("MONKEYS"),
                       vector.getTable()->getMetadata()->get<int>("MONKEYS") );
    BOOST_CHECK_EQUAL( readVector.getTable()->getMetadata()->nameCount(),
                       vector.getTable()->getMetadata()->nameCount() );

    ExtractSchemaStrings func1;  schema.forEach(func1);
    ExtractSchemaStrings func2;  readVector.getSchema().forEach(func2);
    BOOST_CHECK( func1.names == func2.names );
    BOOST_CHECK( func1.docs == func2.docs );
    BOOST_CHECK( func1.units == func2.units );

    {
        SourceRecord const & a1 = vector[0];
        SourceRecord const & b1 = readVector[0];
        BOOST_CHECK_EQUAL( a1.get(a_b_i), b1.get(a_b_i) );
        BOOST_CHECK_EQUAL( a1.get(a_b_i_valid), b1.get(a_b_i_valid) );
        BOOST_CHECK_CLOSE_FRACTION( a1.get(a_c_f), b1.get(a_c_f), 1E-8 );
        BOOST_CHECK_CLOSE_FRACTION( a1.get(e_g_d), b1.get(e_g_d), 1E-16 );
        BOOST_CHECK_EQUAL( a1.get(e_g_d_flag1), b1.get(e_g_d_flag1) );
        BOOST_CHECK_EQUAL( a1.get(e_g_d_flag2), b1.get(e_g_d_flag2) );
        BOOST_CHECK_EQUAL( a1.get(a_s), b1.get(a_s) );
        BOOST_CHECK_CLOSE_FRACTION( a1.get(a_b_p.getX()), b1.get(a_b_p.getX()), 1E-8 );
        BOOST_CHECK_CLOSE_FRACTION( a1.get(a_b_p.getY()), b1.get(a_b_p.getY()), 1E-8 );
        Footprint const & fp1a = *a1.getFootprint();
        Footprint const & fp1b = *b1.getFootprint();
        BOOST_CHECK( std::equal(fp1a.getSpans()->begin(), fp1a.getSpans()->end(), fp1b.getSpans()->begin(),
                                EqualityCompare()) );
        BOOST_CHECK( std::equal(fp1a.getPeaks().begin(), fp1a.getPeaks().end(), fp1b.getPeaks().begin(),
                                EqualityCompare()) );
        BOOST_CHECK_EQUAL( fp1a.getBBox(), fp1b.getBBox() );

        SourceRecord const & a2 = vector[1];
        SourceRecord const & b2 = readVector[1];
        BOOST_CHECK_EQUAL( a2.get(a_b_i), b2.get(a_b_i) );
        BOOST_CHECK_EQUAL( a2.get(a_b_i_valid), b2.get(a_b_i_valid) );
        BOOST_CHECK_CLOSE_FRACTION( a2.get(a_c_f), b2.get(a_c_f), 1E-8 );
        BOOST_CHECK_CLOSE_FRACTION( a2.get(e_g_d), b2.get(e_g_d), 1E-16 );
        BOOST_CHECK_EQUAL( a2.get(e_g_d_flag1), b2.get(e_g_d_flag1) );
        BOOST_CHECK_EQUAL( a2.get(e_g_d_flag2), b2.get(e_g_d_flag2) );
        BOOST_CHECK_EQUAL( a2.get(a_s), b2.get(a_s) );
        BOOST_CHECK_CLOSE_FRACTION( a2.get(a_b_p.getX()), b2.get(a_b_p.getX()), 1E-8 );
        BOOST_CHECK_CLOSE_FRACTION( a2.get(a_b_p.getY()), b2.get(a_b_p.getY()), 1E-8 );
        Footprint const & fp2a = *a2.getFootprint();
        Footprint const & fp2b = *b2.getFootprint();
        BOOST_CHECK( std::equal(fp2a.getSpans()->begin(), fp2a.getSpans()->end(), fp2b.getSpans()->begin(),
                                EqualityCompare()) );
        BOOST_CHECK( std::equal(fp2a.getPeaks().begin(), fp2a.getPeaks().end(), fp2b.getPeaks().begin(),
                                EqualityCompare()) );
        BOOST_CHECK_EQUAL( fp2a.getBBox(), fp2b.getBBox() );
    }

    boost::filesystem::remove(filename);

}

BOOST_AUTO_TEST_CASE(ticket2164) {
    using namespace lsst::afw::table;
    Schema schema;
    schema.addField<int>("i", "test int field");
    schema.addField<double>("d", "test double field");
    BaseCatalog cat(schema);
    cat.addNew();
    cat.addNew();
    ConstBaseCatalog constCat(cat);
    BOOST_CHECK_THROW( constCat.getColumnView(), lsst::pex::exceptions::LogicError );

}
