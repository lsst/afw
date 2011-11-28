#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE catalog-table
#include "boost/test/unit_test.hpp"

#include <iostream>
#include <iterator>
#include <algorithm>

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/SimpleTable.h"

BOOST_AUTO_TEST_CASE(testIterators) {
    using namespace lsst::afw::table;

    Layout layout;
    Key<double> key = layout.add(Field<double>("f", "doc"));
    SimpleTable table(layout, 40);

    /*
     *  top:              ------ 1 ------                ------ 2 ------                ------ 3 ------
     *                  /        |        \            /        |        \            /        |        \ 
     *  middle:        4         5         6          7         8         9          10       11        12
     *              /  |  \   /  |  \   /  |  \    /  |  \   /  |  \   /  |  \    /  |  \   /  |  \   /  |  \
     *  bottom:    13 14 15  16 17 18  19 20 21   22 23 24  25 26 27  28 29 30   31 32 33  34 35 36  37 38 39
     */
    std::list<SimpleRecord> top;
    for (int i = 0; i < 3; ++i) {
        top.push_back(table.addRecord());
        top.back()[key] = Eigen::ArrayXd::Random(1)[0];
    }
    std::list<SimpleRecord> middle;
    for (std::list<SimpleRecord>::iterator i = top.begin(); i != top.end(); ++i) {
        for (int j = 0; j < 3; ++j) {
            middle.push_back(i->addChild());
            middle.back()[key] = Eigen::ArrayXd::Random(1)[0];
        }
    }
    std::list<SimpleRecord> bottom;
    for (std::list<SimpleRecord>::iterator j = middle.begin(); j != middle.end(); ++j) {
        for (int k = 0; k < 3; ++k) {
            bottom.push_back(j->addChild());
            bottom.back()[key] = Eigen::ArrayXd::Random(1)[0];
        }
    }

    // Test set-like iterators on table itself; should be ordered by ID.
    {
        RecordId n = 1;
        for (SimpleTable::Iterator i = table.begin(); i != table.end(); ++i, ++n) {
            BOOST_CHECK_EQUAL(i->getId(), n);
        }
        BOOST_CHECK_EQUAL(n, 40ul);
    }

    // Test tree iterators with NO_CHILDREN; should be equivalent to "top".
    {
        RecordId n = 1;
        SimpleTable::Tree tree = table.asTree(NO_CHILDREN);
        for (SimpleTable::Tree::Iterator i = tree.begin(); i != tree.end(); ++i, ++n) {
            BOOST_CHECK_EQUAL(i->getId(), n);
        }
        BOOST_CHECK_EQUAL(n, 4ul);
    }

    // Test tree iterators with ALL_RECORDS and child iteration; should be depth-first search.
    {
        RecordId order[] = {
            1,  4, 13, 14, 15,  5, 16, 17, 18,  6, 19, 20, 21, 
            2,  7, 22, 23, 24,  8, 25, 26, 27,  9, 28, 29, 30,
            3, 10, 31, 32, 33, 11, 34, 35, 36, 12, 37, 38, 39
        };
        int n = 0;
        SimpleTable::Tree tree = table.asTree(ALL_RECORDS);
        SimpleTable::Tree::Iterator t = tree.begin();
        for (std::list<SimpleRecord>::iterator i = top.begin(); i != top.end(); ++i) {
            BOOST_CHECK_EQUAL(t->getId(), i->getId());
            BOOST_CHECK_EQUAL(t->getId(), order[n]);
            ++t, ++n;
            SimpleRecord::Children ic = i->getChildren(NO_CHILDREN);
            for (SimpleRecord::Children::Iterator j = ic.begin(); j != ic.end(); ++j) {
                BOOST_CHECK_EQUAL(t->getId(), j->getId());
                BOOST_CHECK_EQUAL(t->getId(), order[n]);
                ++t, ++n;
                SimpleRecord::Children jc = j->getChildren(NO_CHILDREN);
                for (SimpleRecord::Children::Iterator k = jc.begin(); k != jc.end(); ++k) {
                    BOOST_CHECK_EQUAL(t->getId(), k->getId());
                    BOOST_CHECK_EQUAL(t->getId(), order[n]);
                    ++t, ++n;
                }
            }
        }
        BOOST_CHECK( t == tree.end() );
    }
}

BOOST_AUTO_TEST_CASE(testSimpleTable) {

    using namespace lsst::afw::table;

    Layout layout;
    
    Key<int> myInt = layout.add(Field< int >("myIntField", "an integer scalar field."));
    
    Key< Array<double> > myArray 
        = layout.add(Field< Array<double> >("myArrayField", "a double array field.", 5));
    
    layout.add(Field< float >("myFloatField", "a float scalar field."));

    Key<float> myFloat = layout.find<float>("myFloatField").key;

    Layout::Description description = layout.describe();

    std::ostream_iterator<FieldDescription> osi(std::cout, "\n");
    std::copy(description.begin(), description.end(), osi);
    
    SimpleTable table(layout, 16);
    
    SimpleRecord r1 = table.addRecord();
    BOOST_CHECK_EQUAL(r1.getId(), 1u);
    r1.set(myInt, 53);
    r1.set(myArray, Eigen::ArrayXd::Ones(5));
    r1.set(myFloat, 3.14f);
    BOOST_CHECK_EQUAL(r1.get(myInt), 53);
    BOOST_CHECK((r1.get(myArray) == Eigen::ArrayXd::Ones(5)).all());
    BOOST_CHECK_EQUAL(r1.get(myFloat), 3.14f);

    SimpleRecord r2 = table.addRecord();
    BOOST_CHECK_EQUAL(r2.getId(), 2u);
    BOOST_CHECK_EQUAL(table.getRecordCount(), 2);
    r2.set(myInt, 25);
    r2.set(myFloat, 5.7f);
    r2.set(myArray, Eigen::ArrayXd::Random(5));

    SimpleRecord r1a = *table.begin();
    BOOST_CHECK_EQUAL(r1a.getId(), 1u);
    BOOST_CHECK_EQUAL(r1.get(myInt), r1a.get(myInt));
    BOOST_CHECK_EQUAL(r1.get(myFloat), r1a.get(myFloat));

    BOOST_CHECK(table.isConsolidated());

#if 0
    table.unlink(0);
    BOOST_CHECK(!table.isConsolidated());
    BOOST_CHECK_EQUAL(table.getRecordCount(), 1);

    SimpleRecord r2a = table[0];
    BOOST_CHECK((r2.get(myArray) == r2a.get(myArray)).all());
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
