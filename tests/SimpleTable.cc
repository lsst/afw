#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE catalog-table
#include "boost/test/unit_test.hpp"

#include <iostream>
#include <iterator>
#include <algorithm>

#include "lsst/catalog/Layout.h"
#include "lsst/catalog/SimpleTable.h"

BOOST_AUTO_TEST_CASE(testSimpleTable) {

    using namespace lsst::catalog;

    LayoutBuilder builder;
    
    Key<int> myInt = builder.add(Field< int >("myIntField", "an integer scalar field."));
    
    Key< Array<double> > myArray 
        = builder.add(Field< Array<double> >(5, "myArrayField", "a double array field.", NOT_NULL));
    
    builder.add(Field< float >("myFloatField", "a float scalar field.", NOT_NULL));

    Layout layout = builder.finish();

    Key<float> myFloat = layout.find<float>("myFloatField");

    Layout::Description description = layout.describe();

    std::ostream_iterator<FieldDescription> osi(std::cout, "\n");
    std::copy(description.begin(), description.end(), osi);
    
    SimpleTable table(layout, 16);
    
    SimpleRecord r0 = table.append();

    BOOST_CHECK(r0.isNull(myInt));
    BOOST_CHECK(!r0.isNull(myFloat));

    r0.set(myInt, 53);
    r0.set(myArray, Eigen::VectorXd::Ones(5));
    r0.set(myFloat, 3.14f);

    BOOST_CHECK(!r0.isNull(myInt));
    BOOST_CHECK(!r0.isNull(myFloat));

    BOOST_CHECK_EQUAL(r0.get(myInt), 53);
    BOOST_CHECK((r0.get(myArray) == Eigen::ArrayXd::Ones(5)).all());
    BOOST_CHECK_EQUAL(r0.get(myFloat), 3.14f);

    SimpleRecord r1 = table.append();
    BOOST_CHECK_EQUAL(table.getRecordCount(), 2);
    r1.set(myInt, 25);
    r1.set(myFloat, 5.7f);
    r1.set(myArray, Eigen::VectorXd::Random(5));
    
    r1.unset(myInt);
    r1.unset(myFloat);
    BOOST_CHECK(r1.isNull(myInt));
    BOOST_CHECK(!r1.isNull(myFloat));

    SimpleRecord r0a = table[0];
    BOOST_CHECK_EQUAL(r0.get(myInt), r0a.get(myInt));
    BOOST_CHECK_EQUAL(r0.get(myFloat), r0a.get(myFloat));

    BOOST_CHECK(table.isConsolidated());

    table.erase(0);
    BOOST_CHECK(!table.isConsolidated());
    BOOST_CHECK_EQUAL(table.getRecordCount(), 1);

    SimpleRecord r1a = table[0];
    BOOST_CHECK((r1.get(myArray) == r1a.get(myArray)).all());
}
