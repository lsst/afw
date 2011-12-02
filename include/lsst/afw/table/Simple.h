// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Simple_h_INCLUDED
#define AFW_TABLE_Simple_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "lsst/afw/table/RecordInterface.h"
#include "lsst/afw/table/TableInterface.h"

namespace lsst { namespace afw { namespace table {

class SimpleRecord;
class SimpleTable;

struct Simple {
    typedef SimpleRecord Record;
    typedef SimpleTable Table;
};

class SimpleRecord : public RecordInterface<Simple> {
public:

    SimpleRecord addChild() const { return _addChild(); }

    SimpleRecord addChild(RecordId id) const { return _addChild(id); }

private:

    friend class detail::Access;

    SimpleRecord(detail::RecordBase const & other) : RecordInterface<Simple>(other) {}
};

class SimpleTable : public TableInterface<Simple> {
public:

    SimpleTable(
        Layout const & layout,
        int defaultBlockRecordCount,
        int capacity = 0,
        IdFactory::Ptr const & idFactory = IdFactory::Ptr()
    ) : TableInterface<Simple>(layout, defaultBlockRecordCount, capacity, idFactory) {}

    
    Record addRecord() const { return _addRecord(); }
    Record addRecord(RecordId id) const { return _addRecord(id); }

};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Simple_h_INCLUDED
