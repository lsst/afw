#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE table - aliases
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop

#include <memory>

#include "lsst/utils/tests.h"
#include "lsst/afw/table/AliasMap.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/BaseRecord.h"

// Need to do some of these tests in C++ to subclass BaseTable and use the full functionality of Citizen.

namespace {

class TestTable;

class TestRecord : public lsst::afw::table::BaseRecord {
public:
    explicit TestRecord(ConstructionToken const & token, lsst::afw::table::detail::RecordData && data) :
        lsst::afw::table::BaseRecord(token, std::move(data))
    {}
};

class TestTable : public lsst::afw::table::BaseTable {
public:
    mutable std::string lastAliasChanged;

    static std::shared_ptr<TestTable> make(lsst::afw::table::Schema const& schema) {
        return std::make_shared<TestTable>(schema);
    }

    explicit TestTable(lsst::afw::table::Schema const& schema) : lsst::afw::table::BaseTable(schema) {}

    TestTable(TestTable const& other) : lsst::afw::table::BaseTable(other) {}

protected:
    // Implementing this is the whole reason we made these test classes; want to verify that this gets
    // called at the right times.
    virtual void handleAliasChange(std::string const& alias) { lastAliasChanged = alias; }

    virtual std::shared_ptr<lsst::afw::table::BaseTable> _clone() const {
        return std::make_shared<TestTable>(*this);
    }

    virtual std::shared_ptr<lsst::afw::table::BaseRecord> _makeRecord() {
        return constructRecord<TestRecord>();
    }
};

}  // namespace

BOOST_AUTO_TEST_CASE(aliasMapLinks) {
    lsst::afw::table::Schema schema;
    schema.addField<int>("a", "doc for a");
    schema.getAliasMap()->set("b", "a");
    std::shared_ptr<TestTable> table = TestTable::make(schema);

    // Should have deep-copied the AliasMap, so pointers should not be equal
    std::shared_ptr<lsst::afw::table::AliasMap> aliases = table->getSchema().getAliasMap();
    BOOST_CHECK(schema.getAliasMap() != aliases);

    // If we set an alias in the map attached to the table, the table should be notified
    aliases->set("c", "a");
    BOOST_CHECK_EQUAL(table->lastAliasChanged, "c");

    // Now we delete the table, and then verify that the link to the table has been broken
    table.reset();

    // If the link isn't broken, this will segfault.
    aliases->set("d", "a");
}

BOOST_AUTO_TEST_CASE(Hash) {
    lsst::utils::assertValidHash<lsst::afw::table::AliasMap>();

    lsst::afw::table::AliasMap map1, map2;
    lsst::afw::table::Schema schema;
    schema.addField<int>("a", "doc for a");

    map1.set("b", "a");
    map2.set("b", "a");
    schema.getAliasMap()->set("b", "a");
    std::shared_ptr<TestTable> table = TestTable::make(schema);
    lsst::afw::table::AliasMap map3 = *(table->getSchema().getAliasMap());

    lsst::utils::assertHashesEqual(map1, map2);
    lsst::utils::assertHashesEqual(map2, map3);
}
