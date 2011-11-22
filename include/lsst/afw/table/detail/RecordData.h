// -*- c++ -*-
#ifndef AFW_TABLE_DETAIL_RecordData_h_INCLUDED
#define AFW_TABLE_DETAIL_RecordData_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "boost/shared_ptr.hpp"

namespace lsst { namespace afw { namespace table {

namespace detail {

enum IteratorTypeEnum { NO_CHILDREN, ALL_RECORDS };

class RecordAux {
public:
    typedef boost::shared_ptr<RecordAux> Ptr;
    virtual ~RecordAux() {}
};

struct RecordData {
    
    RecordId id;
    RecordAux::Ptr aux;
    RecordData * parent;
    RecordData * child;
    RecordData * sibling;

    RecordData() : id(0), aux(), parent(0), child(0), sibling(0) {}
};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_RecordData_h_INCLUDED
