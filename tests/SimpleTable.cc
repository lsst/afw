#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE catalog-table
#include "boost/test/unit_test.hpp"

#include <iostream>
#include <iterator>
#include <algorithm>

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/SimpleTable.h"

BOOST_AUTO_TEST_CASE(testSimpleTable) {

    using namespace lsst::afw::table;

    LayoutBuilder builder;
    
    Key<int> myInt = builder.add(Field< int >("myIntField", "an integer scalar field."));
    
    Key< Array<double> > myArray 
        = builder.add(Field< Array<double> >("myArrayField", "a double array field.", 5));
    
    builder.add(Field< float >("myFloatField", "a float scalar field."));

    Layout layout = builder.finish();

    Key<float> myFloat = layout.find<float>("myFloatField").key;

    Layout::Description description = layout.describe();

    std::ostream_iterator<FieldDescription> osi(std::cout, "\n");
    std::copy(description.begin(), description.end(), osi);
    
    SimpleTable table(layout, 16);
    
    SimpleRecord r0 = table.addRecord();

    r0.set(myInt, 53);
    r0.set(myArray, Eigen::ArrayXd::Ones(5));
    r0.set(myFloat, 3.14f);

    BOOST_CHECK_EQUAL(r0.get(myInt), 53);
    BOOST_CHECK((r0.get(myArray) == Eigen::ArrayXd::Ones(5)).all());
    BOOST_CHECK_EQUAL(r0.get(myFloat), 3.14f);

    SimpleRecord r1 = table.addRecord();
    BOOST_CHECK_EQUAL(table.getRecordCount(), 2);
    r1.set(myInt, 25);
    r1.set(myFloat, 5.7f);
    r1.set(myArray, Eigen::ArrayXd::Random(5));

    SimpleRecord r0a = table.front();
    BOOST_CHECK_EQUAL(r0.get(myInt), r0a.get(myInt));
    BOOST_CHECK_EQUAL(r0.get(myFloat), r0a.get(myFloat));

    BOOST_CHECK(table.isConsolidated());

#if 0
    table.erase(0);
    BOOST_CHECK(!table.isConsolidated());
    BOOST_CHECK_EQUAL(table.getRecordCount(), 1);

    SimpleRecord r1a = table[0];
    BOOST_CHECK((r1.get(myArray) == r1a.get(myArray)).all());
#endif
}

#if 0

BOOST_AUTO_TEST_CASE(testColumnView) {

    using namespace lsst::afw::table;

    LayoutBuilder builder;
    Key<float> floatKey = builder.add(Field<float>("f1", "f1 doc"));
    Key< Array<double> > arrayKey = builder.add(Field< Array<double> >("f2", "f2 doc", 5));
    Layout layout = builder.finish();
    
    SimpleTable table(layout, 16);
    Eigen::ArrayXd r = Eigen::ArrayXd::Random(20);
    for (int i = 0; i < 20; ++i) {
        SimpleRecord record = table.addRecord();
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
            BOOST_CHECK_EQUAL(record.get(arrayKey)[j], recordCopy.get(arrayKey[j]));
            BOOST_CHECK_EQUAL(record.get(arrayKey)[j], columns[arrayKey[j]][i]);
            BOOST_CHECK_EQUAL(record.get(arrayKey)[j], columns[arrayKey][i][j]);
        }
    }
    
}

#endif
