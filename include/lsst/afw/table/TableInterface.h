// -*- c++ -*-
#ifndef AFW_TABLE_TableInterface_h_INCLUDED
#define AFW_TABLE_TableInterface_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "boost/iterator/transform_iterator.hpp"

#include "lsst/base.h"
#include "lsst/afw/table/RecordInterface.h"
#include "lsst/afw/table/detail/TableBase.h"
#include "lsst/afw/table/detail/TreeIteratorBase.h"
#include "lsst/afw/table/detail/SetIteratorBase.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

template <typename RecordT>
class TreeView : private detail::TableBase {
public:

    typedef RecordT Record;
    typedef boost::transform_iterator<detail::RecordConverter<RecordT>,detail::TreeIteratorBase> Iterator;
    typedef Iterator iterator;
    typedef Iterator const_iterator;

    Iterator begin() const {
        return Iterator(this->_beginTree(_mode), detail::RecordConverter<RecordT>());
    }

    Iterator end() const {
        return Iterator(this->_endTree(_mode), detail::RecordConverter<RecordT>());
    }

    Iterator erase(Iterator const & iter) {
        return Iterator(this->_erase(iter), detail::RecordConverter<RecordT>());
    }

private:

    template <typename OtherRecordT, typename TableAuxT> friend class TableInterface;

    TreeView(detail::TableBase const & table, IteratorMode mode) : detail::TableBase(table), _mode(mode) {}

    IteratorMode _mode;
};

template <typename RecordT, typename TableAuxT=AuxBase>
class TableInterface : public detail::TableBase {
public:

    typedef RecordT Record;
    typedef TreeView<Record> Tree;
    typedef boost::transform_iterator<detail::RecordConverter<RecordT>,detail::SetIteratorBase> Iterator;
    typedef Iterator iterator;
    typedef Iterator const_iterator;

    Tree asTree(IteratorMode mode) const { return Tree(*this, mode); }

    Iterator begin() const {
        return Iterator(this->_beginSet(), detail::RecordConverter<RecordT>());
    }

    Iterator end() const {
        return Iterator(this->_endSet(), detail::RecordConverter<RecordT>());
    }

    void erase(Record const & record) { this->_erase(record); }

    Iterator erase(Iterator const & iter) {
        return Iterator(this->_erase(iter.base()), detail::RecordConverter<RecordT>());
    }

    Record operator[](RecordId id) const {
        return detail::Access::makeRecord<RecordT>(this->_get(id));
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
