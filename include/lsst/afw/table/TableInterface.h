// -*- c++ -*-
#ifndef AFW_TABLE_TableInterface_h_INCLUDED
#define AFW_TABLE_TableInterface_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "boost/iterator/transform_iterator.hpp"

#include "lsst/base.h"
#include "lsst/afw/table/RecordInterface.h"
#include "lsst/afw/table/detail/TableBase.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

template <typename RecordT, typename TableAuxT=AuxBase>
class TableInterface : public detail::TableBase {
public:

    typedef RecordT Record;
    typedef boost::transform_iterator<detail::RecordConverter<RecordT>,detail::IteratorBase> Iterator;

    Iterator begin(IteratorMode mode = ALL_RECORDS) const {
        return Iterator(detail::RecordConverter<RecordT>(), this->_begin(mode));
    }

    Iterator end(IteratorMode mode = ALL_RECORDS) const {
        return Iterator(detail::RecordConverter<RecordT>(), this->_end(mode));
    }

    Record front() const {
        return detail::Access::makeRecord<Record>(this->_front());
    }

    Record back(IteratorMode mode = ALL_RECORDS) const {
        return detail::Access::makeRecord<Record>(this->_back(mode));
    }

protected:

    typedef TableAuxT TableAux;
    typedef typename Record::RecordAux RecordAux;

    TableInterface(
        Layout const & layout,
        int defaultBlockRecordCount,
        int capacity,
        IdFactory::Ptr const & idFactory = IdFactory::Ptr(),
        PTR(TableAux) const & aux = PTR(TableAux)()
    ) : detail::TableBase(layout, defaultBlockRecordCount, capacity, idFactory, aux) {}

    Record _addRecord(RecordId id, PTR(RecordAux) const & aux = PTR(RecordAux)()) {
        return detail::Access::makeRecord<Record>(this->detail::TableBase::_addRecord(id, aux));
    }

    Record _addRecord(PTR(RecordAux) const & aux = PTR(RecordAux)()) {
        return detail::Access::makeRecord<Record>(this->detail::TableBase::_addRecord(aux));
    }

    PTR(TableAux) getAux() const {
        return boost::static_pointer_cast<TableAux>(detail::TableBase::getAux());
    }
    
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_TableInterface_h_INCLUDED
