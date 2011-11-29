// -*- lsst-c++ -*-
#ifndef AFW_TABLE_SimpleRecord_h_INCLUDED
#define AFW_TABLE_SimpleRecord_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "lsst/afw/table/RecordInterface.h"

namespace lsst { namespace afw { namespace table {

class SimpleRecord : public RecordInterface<SimpleRecord> {
public:

    SimpleRecord addChild() const { return _addChild(); }

    SimpleRecord addChild(RecordId id) const { return _addChild(id); }

private:

    friend class detail::Access;

    SimpleRecord(detail::RecordBase const & other) : RecordInterface<SimpleRecord>(other) {}
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_SimpleRecord_h_INCLUDED
