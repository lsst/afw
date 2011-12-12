// -*- lsst-c++ -*-
#ifndef AFW_TABLE_DETAIL_RecordData_h_INCLUDED
#define AFW_TABLE_DETAIL_RecordData_h_INCLUDED

#include "boost/shared_ptr.hpp"
#include "boost/intrusive/set.hpp"

#include "lsst/base.h"
#include "lsst/afw/table/misc.h"

namespace lsst { namespace afw { namespace table { namespace detail {

struct RecordData : public boost::intrusive::set_base_hook<> {

    struct Links {
        RecordData * parent;
        RecordData * child;
        RecordData * previous;
        RecordData * next;

        void initialize() {
            parent = 0;
            child = 0;
            previous = 0;
            next = 0;
        }
    };
    
    RecordId id;
    PTR(AuxBase) aux;
    union {
        Links links;
        RecordId parentId;
    };

    bool operator<(RecordData const & other) const { return id < other.id; }

    RecordData() : id(0), aux() { links.initialize(); }
};

struct CompareRecordIdLess {
    bool operator()(RecordId id, RecordData const & data) const {
        return id < data.id;
    }
    bool operator()(RecordData const & data, RecordId id) const {
        return data.id < id;
    }
};

typedef boost::intrusive::set<RecordData> RecordSet;

void setupPointers(RecordSet & records, RecordData * & back);

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_RecordData_h_INCLUDED
