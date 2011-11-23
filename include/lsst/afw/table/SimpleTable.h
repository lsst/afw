// -*- c++ -*-
#ifndef AFW_TABLE_SimpleTable_h_INCLUDED
#define AFW_TABLE_SimpleTable_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "lsst/afw/table/TableInterface.h"
#include "lsst/afw/table/SimpleRecord.h"

namespace lsst { namespace afw { namespace table {

class SimpleTable : public TableInterface<SimpleRecord> {
public:

    SimpleTable(
        Layout const & layout,
        int defaultBlockRecordCount,
        int capacity = 0,
        IdFactory::Ptr const & idFactory = IdFactory::Ptr()
    ) : TableInterface<SimpleRecord>(layout, defaultBlockRecordCount, capacity, idFactory) {}

    
    Record addRecord() { return _addRecord(); }
    Record addRecord(RecordId id) { return _addRecord(id); }

};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_SimpleTable_h_INCLUDED
