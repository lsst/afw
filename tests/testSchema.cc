#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE table-schema
#include "boost/test/unit_test.hpp"

#include <iostream>
#include <iterator>
#include <algorithm>
#include <map>

#include "lsst/afw/table/Schema.h"

using namespace lsst::afw::table;

BOOST_AUTO_TEST_CASE(testSchema) {

    Schema schema(false);
    Key<int> abi_k = schema.addField<int>("a.b.i", "int");
    Key<float> acf_k = schema.addField<float>("a.c.f", "float");
    Key<double> egd_k = schema.addField<double>("e.g.d", "double");
    Key< Point<float> > abp_k = schema.addField< Point<float> >("a.b.p", "point");
    SchemaItem< Point<float> > abp_si = schema.find< Point<float> >("a.b.p");
    BOOST_CHECK( abp_si.key == abp_k );
    BOOST_CHECK_EQUAL( abp_si.field.getName(), "a.b.p" );
    SchemaItem<float> abpx_si = schema.find<float>("a.b.p.x");
    BOOST_CHECK( abp_k.getX() == abpx_si.key );
    BOOST_CHECK_EQUAL( abpx_si.field.getName(), "a.b.p.x" );
    BOOST_CHECK_EQUAL( abpx_si.field.getDoc(), "point" );
    BOOST_CHECK( abp_k == schema["a.b.p"] );
    BOOST_CHECK( abp_k.getX() == schema["a.b.p.x"] );

    std::set<std::string> names;
    names.insert("a.b.p");
    names.insert("a.b.i");
    names.insert("a.c.f");
    names.insert("e.g.d");
    BOOST_CHECK( schema.getNames() == names );

    names.clear();
    names.insert("a");
    names.insert("e");
    BOOST_CHECK( schema.getNames(true) == names );

    names.clear();
    names.insert("b.i");
    names.insert("b.p");
    names.insert("c.f");
    BOOST_CHECK( schema["a"].getNames() == names );

    names.clear();
    names.insert("b");
    names.insert("c");
    BOOST_CHECK( schema["a"].getNames(true) == names );

    Schema schema2(schema);
    BOOST_CHECK( schema == schema2 );
    schema2.addField<double>("q", "another double");
    BOOST_CHECK( schema != schema2 );

    Schema schema3(false);
    schema3.addField<int>("i", "int");
    schema3.addField<float>("f", "float");
    schema3.addField<double>("d", "double");
    schema3.addField< Point<float> >("p", "point");

    BOOST_CHECK( schema3 == schema );

}
