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

BOOST_AUTO_TEST_CASE(testColumnView) {

    using namespace lsst::catalog;

    LayoutBuilder builder;
    Key<float> floatKey = builder.add(Field<float>("f1", "f1 doc", ALLOW_NULL));
    Key< Array<double> > arrayKey = builder.add(Field< Array<double> >(5, "f2", "f2 doc", ALLOW_NULL));
    Layout layout = builder.finish();
    
    SimpleTable table(layout, 16);
    Eigen::ArrayXd r = Eigen::ArrayXd::Random(20);
    for (int i = 0; i < 20; ++i) {
        SimpleRecord record = table.append();
        record.set(floatKey, r[i]);
        record.set(arrayKey, Eigen::ArrayXd::Random(5));
        if (i < 16) {
            BOOST_CHECK(table.isConsolidated());
        } else {
            BOOST_CHECK(!table.isConsolidated());
        }
    }
    
    SimpleTable tableCopy(table);
    ColumnView columns = table.consolidate();
    BOOST_CHECK(!tableCopy.isConsolidated());
    BOOST_CHECK(table.isConsolidated());

    for (int i = 0; i < 20; ++i) {
        SimpleRecord record = table[i];
        SimpleRecord recordCopy = tableCopy[i];
        BOOST_CHECK_EQUAL(record.get(floatKey), recordCopy.get(floatKey));
        BOOST_CHECK_EQUAL(record.get(floatKey), columns[floatKey][i]);
        for (int j = 0; j < 5; ++j) {
            BOOST_CHECK_EQUAL(record.get(arrayKey)[j], recordCopy.get(arrayKey)[j]);
            BOOST_CHECK_EQUAL(record.get(arrayKey)[j], recordCopy.get(arrayKey.at(j)));
            BOOST_CHECK_EQUAL(record.get(arrayKey)[j], columns[arrayKey.at(j)][i]);
        }
    }
    
}
