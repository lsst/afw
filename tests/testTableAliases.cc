#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE table-aliases
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop

#include "boost/scoped_ptr.hpp"

#include "lsst/afw/table/AliasMap.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/BaseRecord.h"

// Need to do some of these tests in C++ to subclass BaseTable and use the full functionality of Citizen.

namespace {

class TestTable;

class TestRecord : public lsst::afw::table::BaseRecord {
public:

    explicit TestRecord(PTR(TestTable) const & table);

};

class TestTable : public lsst::afw::table::BaseTable {
public:

    mutable std::string lastAliasChanged;

    static PTR(TestTable) make(lsst::afw::table::Schema const & schema) {
        return boost::make_shared<TestTable>(schema);
    }

    explicit TestTable(lsst::afw::table::Schema const & schema) : lsst::afw::table::BaseTable(schema) {}

    TestTable(TestTable const & other) : lsst::afw::table::BaseTable(other) {}

protected:

    // Implementing this is the whole reason we made these test classes; want to verify that this gets
    // called at the right times.
    virtual void handleAliasChange(std::string const & alias) { lastAliasChanged = alias; }

    virtual PTR(lsst::afw::table::BaseTable) _clone() const { return boost::make_shared<TestTable>(*this); }

    virtual PTR(lsst::afw::table::BaseRecord) _makeRecord() {
        return boost::make_shared<TestRecord>(getSelf<TestTable>());
    }

};

TestRecord::TestRecord(PTR(TestTable) const & table) : lsst::afw::table::BaseRecord(table) {}

} // anonymous

BOOST_AUTO_TEST_CASE(aliasMapLinks) {
    lsst::afw::table::Schema schema;
    schema.addField<int>("a", "doc for a");
    schema.getAliasMap()->set("b", "a");
    PTR(TestTable) table = TestTable::make(schema);

    // Should have deep-copied the AliasMap, so pointers should not be equal
    PTR(lsst::afw::table::AliasMap) aliases = table->getSchema().getAliasMap();
    BOOST_CHECK(schema.getAliasMap() != aliases);

    // If we set an alias in the map attached to the table, the table should be notified
    aliases->set("c", "a");
    BOOST_CHECK_EQUAL(table->lastAliasChanged, "c");

    // Now we delete the table, and then verify that the link to the table has been broken
    lsst::daf::base::Citizen::memId tableCitizenId = table->getId();
    table.reset();

    // Is it really dead?
    typedef std::vector<lsst::daf::base::Citizen const*> CensusVector;
    std::unique_ptr<CensusVector const> census(lsst::daf::base::Citizen::census());
    for (CensusVector::const_iterator i = census->begin(); i != census->end(); ++i) {
        BOOST_CHECK((**i).getId() != tableCitizenId);
    }

    // If the link isn't broken, this will segfault.
    aliases->set("d", "a");
}
