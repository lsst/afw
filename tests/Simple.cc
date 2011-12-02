#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE catalog-table
#include "boost/test/unit_test.hpp"

#include <iostream>
#include <iterator>
#include <algorithm>

#include "boost/assign/std/list.hpp"

#include "lsst/afw/table/Layout.h"
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

    Example() : layout(), key(layout.add(Field<double>("f", "doc"))), table(layout, 10) {
        std::list<SimpleRecord> top;
        std::list<SimpleRecord> middle;
        std::list<SimpleRecord> bottom;
        for (int i = 0; i < 3; ++i) {
            top.push_back(table.addRecord());
            top.back()[key] = Eigen::ArrayXd::Random(1)[0];
            tableOrder.push_back(top.back().getId());
            treeOrder[NO_NESTING].push_back(top.back().getId());
        }
        for (std::list<SimpleRecord>::iterator i = top.begin(); i != top.end(); ++i) {
            for (int j = 0; j < 3; ++j) {
                middle.push_back(i->addChild());
                middle.back()[key] = Eigen::ArrayXd::Random(1)[0];
                tableOrder.push_back(middle.back().getId());
            }
        }
        for (std::list<SimpleRecord>::iterator j = middle.begin(); j != middle.end(); ++j) {
            for (int k = 0; k < 3; ++k) {
                bottom.push_back(j->addChild());
                bottom.back()[key] = Eigen::ArrayXd::Random(1)[0];
                tableOrder.push_back(bottom.back().getId());
            }
        }
        using namespace boost::assign;
        treeOrder[DEPTH_FIRST] +=   // using Boost.Assign here
            1,  4, 13, 14, 15,  5, 16, 17, 18,  6, 19, 20, 21, 
            2,  7, 22, 23, 24,  8, 25, 26, 27,  9, 28, 29, 30,
            3, 10, 31, 32, 33, 11, 34, 35, 36, 12, 37, 38, 39;
    }

    template <typename Container>
    static void _checkIteration(Container const & container, std::list<RecordId> const & order) {
        std::list<RecordId>::const_iterator o = order.begin();
        for (typename Container::Iterator i = container.begin(); i != container.end(); ++i, ++o) {
            BOOST_CHECK_EQUAL(i->getId(), *o);
            if (i->hasChildren()) {
                BOOST_CHECK_THROW(i->unlink(), lsst::pex::exceptions::LogicErrorException);
                BOOST_CHECK_THROW(container.unlink(i), lsst::pex::exceptions::LogicErrorException);
            }
        }
        BOOST_CHECK( o == order.end() );
    }

    void checkIteration() const {
        _checkIteration(table, tableOrder);
        _checkIteration(table.asTree(NO_NESTING), treeOrder[NO_NESTING]);
        _checkIteration(table.asTree(DEPTH_FIRST), treeOrder[DEPTH_FIRST]);
    }
    
    void remove(RecordId id) {
        tableOrder.remove(id);
        treeOrder[NO_NESTING].remove(id);
        treeOrder[DEPTH_FIRST].remove(id);
    }

    
    Layout layout;
    Key<double> key;
    SimpleTable table;
    std::list<RecordId> tableOrder;
    std::list<RecordId> treeOrder[2];
};

BOOST_AUTO_TEST_CASE(testIterators) {

    Example example;
    example.checkIteration();

    // Test record child iterators.
    {
        SimpleTable::Tree tree = example.table.asTree(DEPTH_FIRST);
        SimpleTable::Tree top = example.table.asTree(NO_NESTING);
        SimpleTable::Tree::Iterator t = tree.begin();
        for (SimpleTable::Tree::Iterator i = top.begin(); i != top.end(); ++i) {
            BOOST_CHECK_EQUAL(t->getId(), i->getId());
            BOOST_CHECK_THROW(t->unlink(), lsst::pex::exceptions::LogicErrorException);
            BOOST_CHECK_THROW(tree.unlink(t), lsst::pex::exceptions::LogicErrorException);
            ++t;
            SimpleRecord::Children ic = i->getChildren(NO_NESTING);
            for (SimpleRecord::Children::Iterator j = ic.begin(); j != ic.end(); ++j) {
                BOOST_CHECK_EQUAL(t->getId(), j->getId());
                BOOST_CHECK_THROW(t->unlink(), lsst::pex::exceptions::LogicErrorException);
                BOOST_CHECK_THROW(tree.unlink(t), lsst::pex::exceptions::LogicErrorException);
                ++t;
                SimpleRecord::Children jc = j->getChildren(NO_NESTING);
                for (SimpleRecord::Children::Iterator k = jc.begin(); k != jc.end(); ++k) {
                    BOOST_CHECK_EQUAL(t->getId(), k->getId());
                    ++t;
                }
            }
        }
        BOOST_CHECK( t == tree.end() );
    }

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
        BOOST_CHECK_EQUAL(i24->getId(), 24ul);
        example.remove(23);
        example.checkIteration();
    } {
        SimpleRecord r7 = example.table[7];
        BOOST_CHECK(r7.hasChildren());
        SimpleTable::Tree::Iterator i22 = r7.asTreeIterator(DEPTH_FIRST); ++i22;
        BOOST_CHECK_EQUAL(i22->getId(), 22ul);
        SimpleTable::Tree::Iterator i24 = example.table.asTree(DEPTH_FIRST).unlink(i22);
        example.remove(22);
        example.checkIteration();
        BOOST_CHECK_EQUAL(i24->getId(), 24ul);
        BOOST_CHECK_THROW(
            example.table.asTree(NO_NESTING).unlink(i24), 
            lsst::pex::exceptions::LogicErrorException
        );
        {
            SimpleTable::Tree::Iterator i8 = example.table.asTree(DEPTH_FIRST).unlink(i24);
            BOOST_CHECK_EQUAL(i8->getId(), 8ul);
            example.remove(24);
            example.checkIteration();
        } {
            BOOST_CHECK(!r7.hasChildren());
            SimpleTable::Tree::Iterator i8 
                = example.table.asTree(NO_NESTING).unlink(r7.asTreeIterator(NO_NESTING));
            BOOST_CHECK_EQUAL(i8->getId(), 8ul);
            example.remove(7);
            example.checkIteration();
        }
    } {
        SimpleTable::Iterator iEnd = example.table.unlink(example.table.find(39));
        BOOST_CHECK(iEnd == example.table.end());
        example.remove(39);
        example.checkIteration();
    } {
        SimpleRecord r38 = example.table[38];
        SimpleTable::Tree::Iterator iEnd 
            = example.table.asTree(DEPTH_FIRST).unlink(r38.asTreeIterator(DEPTH_FIRST));
        BOOST_CHECK(iEnd == example.table.asTree(DEPTH_FIRST).end());
        example.remove(38);
        example.checkIteration();
        example.table[37].unlink();
        example.remove(37);
        example.checkIteration();
    } {
        SimpleRecord r12 = example.table[12];
        SimpleTable::Tree::Iterator iEnd
            = example.table.asTree(NO_NESTING).unlink(r12.asTreeIterator(NO_NESTING));
        BOOST_CHECK(iEnd == example.table.asTree(NO_NESTING).end());
        example.remove(12);
        example.checkIteration();
    }
}

BOOST_AUTO_TEST_CASE(testConsolidate) {

    Example example;
    example.checkIteration();

    BOOST_CHECK(!example.table.isConsolidated());

    example.table.consolidate();
    example.checkIteration();

    BOOST_CHECK(example.table.isConsolidated());

}

BOOST_AUTO_TEST_CASE(testSimpleTable) {

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

}

#if 0

BOOST_AUTO_TEST_CASE(testColumnView) {

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
