// -*- c++ -*-
#ifndef AFW_TABLE_DETAIL_RecordData_h_INCLUDED
#define AFW_TABLE_DETAIL_RecordData_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "boost/shared_ptr.hpp"

namespace lsst { namespace afw { namespace table {

class AuxBase {
public:
    typedef boost::shared_ptr<AuxBase> Ptr;
    virtual ~AuxBase() {}
};

namespace detail {

struct RecordData {
    
    RecordId id;
    AuxBase::Ptr aux;
    RecordData * parent;
    RecordData * child;
    RecordData * sibling;

    RecordData() : id(0), aux(), parent(0), child(0), sibling(0) {}
};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_RecordData_h_INCLUDED
