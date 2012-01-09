#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE table-simple
#include "boost/test/unit_test.hpp"

#include <iostream>
#include <iterator>
#include <algorithm>
#include <map>

#include "boost/assign/std/list.hpp"

#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/Simple.h"

using namespace lsst::afw::table;

/*
 * A table with the following structure:
 *
 *  top:           ------ 1 ------                ------ 2 ------                ------ 3 ------
 *               /        |        \            /        |        \            /        |        \ 
 *  middle:     4         5         6          7         8         9          10       11        12
 *           /  |  \   /  |  \   /  |  \    /  |  \   /  |  \   /  |  \    /  |  \   /  |  \   /  | \
 *  bottom: 13 14 15  16 17 18  19 20 21   22 23 24  25 26 27  28 29 30   31 32 33  34 35 36  37 38 39
 */
struct Example {

    Example() : schema(true), key(schema.addField<double>("f", "doc")), table(schema) {
        std::list<SimpleRecord> top;
        std::list<SimpleRecord> middle;
        std::list<SimpleRecord> bottom;
        for (int i = 0; i < 3; ++i) {
            top.push_back(table.addRecord());
            top.back()[key] = Eigen::ArrayXd::Random(1)[0];
            values.insert(std::make_pair(top.back().getId(), top.back().get(key)));
            tableOrder.push_back(top.back().getId());
            treeOrder.push_back(top.back().getId());
        }
        for (std::list<SimpleRecord>::iterator i = top.begin(); i != top.end(); ++i) {
            for (int j = 0; j < 3; ++j) {
                middle.push_back(table.addRecord());
                middle.back().setParentId(i->getParentId());
                middle.back()[key] = Eigen::ArrayXd::Random(1)[0];
                values.insert(std::make_pair(middle.back().getId(), middle.back().get(key)));
                tableOrder.push_back(middle.back().getId());
            }
        }
        for (std::list<SimpleRecord>::iterator j = middle.begin(); j != middle.end(); ++j) {
            for (int k = 0; k < 3; ++k) {
                bottom.push_back(table.addRecord());
                bottom.back().setParentId(j->getParentId());
                bottom.back()[key] = Eigen::ArrayXd::Random(1)[0];
                values.insert(std::make_pair(bottom.back().getId(), bottom.back().get(key)));
                tableOrder.push_back(bottom.back().getId());
            }
        }
        using namespace boost::assign;
        treeOrder +=   // using Boost.Assign here
            1,  4, 13, 14, 15,  5, 16, 17, 18,  6, 19, 20, 21, 
            2,  7, 22, 23, 24,  8, 25, 26, 27,  9, 28, 29, 30,
            3, 10, 31, 32, 33, 11, 34, 35, 36, 12, 37, 38, 39;
    }

    void checkChildIteration(
        SimpleRecord const & record,
        std::list<RecordId>::const_iterator & iter
    ) const {
        BOOST_CHECK_EQUAL(record.getId(), *iter);
        ++iter;
        SimpleRecord::Children children = record.getChildren();
        for (SimpleRecord::Children::Iterator i = children.begin(); i != children.end(); ++i) {
            checkChildIteration(*i, iter);
        }
    }

    template <typename Container>
    static void _checkIteration(Container const & container, std::list<RecordId> const & order) {
    }

    void checkIteration() const {
        {
            std::list<RecordId>::const_iterator o = tableOrder.begin();
            for (SimpleTable::Iterator i = table.begin(); i != table.end(); ++i, ++o) {
                BOOST_CHECK_EQUAL(i->getId(), *o);
            }
            BOOST_CHECK( o == tableOrder.end() );
        }
        {
            std::list<RecordId>::const_iterator iter = treeOrder.begin();
            checkChildIteration(table[1], iter);
            checkChildIteration(table[2], iter);
            checkChildIteration(table[3], iter);
        }
    }

    void remove(RecordId id) {
        tableOrder.remove(id);
        treeOrder.remove(id);
    }

    Schema schema;
    Key<double> key;
    SimpleTable table;
    std::list<RecordId> tableOrder;
    std::list<RecordId> treeOrder;
    std::map<RecordId,double> values;
};

BOOST_AUTO_TEST_CASE(testIterators) {

    Example example;
    example.checkIteration();

    // Test all kinds of unlinking in different places, verifying that we haven't messed up iteration.
    {
        SimpleRecord r15 = example.table[15];
        BOOST_CHECK(r15.isLinked());
        r15.unlink();
        BOOST_CHECK(!r15.isLinked());
        BOOST_CHECK(!r15.hasParent());
        BOOST_CHECK(example.table.find(15) == example.table.end());
        BOOST_CHECK_THROW(example.table[15], lsst::pex::exceptions::NotFoundException);
        example.remove(15);
        example.checkIteration();
    }
    {
        SimpleTable::Iterator i23 = example.table.find(23);
        BOOST_CHECK(i23->isLinked());
        SimpleRecord r23 = *i23;
        SimpleTable::Iterator i24 = example.table.unlink(i23);
        BOOST_CHECK(!r23.isLinked());
        BOOST_CHECK(!r23.hasParent());
        BOOST_CHECK_EQUAL(i24->getId(), 24);
        example.remove(23);
        example.checkIteration();
    } {
        SimpleTable::Iterator iEnd = example.table.unlink(example.table.find(39));
        BOOST_CHECK(iEnd == example.table.end());
        example.remove(39);
        example.checkIteration();
    }
}

BOOST_AUTO_TEST_CASE(testConsolidate) {

    TableBase::nRecordsPerBlock = 10;

    Example example;
    example.checkIteration();

    BOOST_CHECK(!example.table.isConsolidated());

    for (SimpleTable::Iterator i = example.table.begin(); i != example.table.end(); ++i) {
        BOOST_CHECK_EQUAL(example.values[i->getId()], i->get(example.key));
    }

    example.table.consolidate();
    example.checkIteration();

    BOOST_CHECK(example.table.isConsolidated());

    for (SimpleTable::Iterator i = example.table.begin(); i != example.table.end(); ++i) {
        BOOST_CHECK_EQUAL(example.values[i->getId()], i->get(example.key));
    }

    TableBase::nRecordsPerBlock = 100;
}

BOOST_AUTO_TEST_CASE(testSimpleTable) {

    Schema schema(false);
    
    Key<int> myInt = schema.addField<int>("myIntField", "an integer scalar field.");
    
    Key< Array<double> > myArray 
        = schema.addField(Field< Array<double> >("myArrayField", "a double array field.", 5));
    
    schema.addField(Field< float >("myFloatField", "a float scalar field."));

    Key<float> myFloat = schema.find<float>("myFloatField").key;
    
    SimpleTable table(schema);
    
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

}

BOOST_AUTO_TEST_CASE(testColumnView) {

    Schema schema(false);
    Key<float> floatKey = schema.addField(Field<float>("f1", "f1 doc"));
    Key< Array<double> > arrayKey = schema.addField(Field< Array<double> >("f2", "f2 doc", 5));
    
    TableBase::nRecordsPerBlock = 16;

    SimpleTable table(schema);
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
    BOOST_CHECK_THROW(table.getColumnView(), lsst::pex::exceptions::LogicErrorException);
    table.consolidate();
    BOOST_CHECK(!tableCopy.isConsolidated());
    BOOST_CHECK(table.isConsolidated());
    ColumnView columns = table.getColumnView();

    SimpleTable::Iterator i1 = table.begin();
    SimpleTable::Iterator i2 = tableCopy.begin();
    for (int n = 0; i1 != table.end(); ++i1, ++i2, ++n) {
        SimpleRecord record = *i1;
        SimpleRecord recordCopy = *i2;
        BOOST_CHECK_EQUAL(record.get(floatKey), recordCopy.get(floatKey));
        BOOST_CHECK_EQUAL(record.get(floatKey), columns[floatKey][n]);
        for (int j = 0; j < 5; ++j) {
            BOOST_CHECK_EQUAL(record.get(arrayKey)[j], recordCopy.get(arrayKey)[j]);
            BOOST_CHECK_EQUAL(record.get(arrayKey)[j], recordCopy.get(arrayKey[j]));
            BOOST_CHECK_EQUAL(record.get(arrayKey)[j], columns[arrayKey[j]][n]);
            BOOST_CHECK_EQUAL(record.get(arrayKey)[j], columns[arrayKey][n][j]);
        }
    }

    TableBase::nRecordsPerBlock = 100;
}

BOOST_AUTO_TEST_CASE(testFlags) {

    Schema schema(false);
    
    int const nFields = 70;
    int const nRecords = 10;

    std::vector< Key<double> > doubleKeys;
    std::vector< Key<Flag> > flagKeys;
    for (int i = 0; i < nFields; ++i) {    
        doubleKeys.push_back(
            schema.addField<double>(
                (boost::format("float%d") % i).str(),
                "an double-precision field."
            )
        );
        flagKeys.push_back(
            schema.addField<Flag>(
                (boost::format("flag%d") % i).str(),
                "a flag field."
            )
        );
    }

    int const nBitsPerStorage = sizeof(Field<Flag>::Element) * 8;
    for (int i = 1; i < nBitsPerStorage; ++i) {
        BOOST_CHECK( flagKeys[i].getStorage() == flagKeys[0].getStorage() );
    }
    for (int i = nBitsPerStorage + 1; i < nRecords; ++i) {
        BOOST_CHECK( flagKeys[i].getStorage() == flagKeys[nBitsPerStorage].getStorage() );
    }

    SimpleTable table(schema, nRecords);
    Eigen::ArrayXXd values = Eigen::ArrayXXd::Random(nRecords, nFields);
    for (int j = 0; j < nRecords; ++j) {
        SimpleRecord r = table.addRecord();
        for (int i = 0; i < nFields; ++i) {
            r[doubleKeys[i]] = values(j, i);
            r.set(flagKeys[i], r[doubleKeys[i]] < 0.5);
        }
        for (int i = 0; i < nFields; ++i) {
            BOOST_CHECK_EQUAL(r.get(flagKeys[i]), r[doubleKeys[i]] < 0.5);
        }
    }
}
