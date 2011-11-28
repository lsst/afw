// -*- c++ -*-
#ifndef AFW_TABLE_DETAIL_RecordData_h_INCLUDED
#define AFW_TABLE_DETAIL_RecordData_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "boost/shared_ptr.hpp"
#include "boost/intrusive/set.hpp"

namespace lsst { namespace afw { namespace table {

class AuxBase {
public:
    typedef boost::shared_ptr<AuxBase> Ptr;
    virtual ~AuxBase() {}
};

namespace detail {

struct RecordData : public boost::intrusive::set_base_hook<> {
    
    RecordId id;
    AuxBase::Ptr aux;
    RecordData * parent;
    RecordData * child;
    RecordData * previous;
    RecordData * next;

    bool operator<(RecordData const & other) const { return id < other.id; }

    RecordData() : id(0), aux(), parent(0), child(0), previous(0), next(0) {}
};

struct CompareRecordIdLess {
    bool operator()(RecordId id, RecordData const & data) const {
        return id < data.id;
    }
    bool operator()(RecordData const & data, RecordId id) const {
        return data.id < id;
    }
};

struct CompareRecordIdEqual {
    bool operator()(RecordId id, RecordData const & data) const {
        return id == data.id;
    }
    bool operator()(RecordData const & data, RecordId id) const {
        return data.id == id;
    }
};

typedef boost::intrusive::set<RecordData> RecordSet;

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_RecordData_h_INCLUDED
