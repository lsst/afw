// -*- c++ -*-
#ifndef AFW_TABLE_TableInterface_h_INCLUDED
#define AFW_TABLE_TableInterface_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "boost/iterator/transform_iterator.hpp"

#include "lsst/base.h"
#include "lsst/afw/table/RecordInterface.h"
#include "lsst/afw/table/detail/TableBase.h"
#include "lsst/afw/table/detail/TreeIteratorBase.h"
#include "lsst/afw/table/detail/IteratorBase.h"
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

    Iterator unlink(Iterator const & iter) const {
        if (iter.base().getMode() != _mode) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LogicErrorException,
                "TreeView and iterator modes do not agree."
            );
        }
        return Iterator(this->_unlink(iter.base()), detail::RecordConverter<RecordT>());
    }

private:

    template <typename OtherRecordT> friend class TableInterface;

    TreeView(detail::TableBase const & table, TreeMode mode) : detail::TableBase(table), _mode(mode) {}

    TreeMode _mode;
};

template <typename RecordT>
class TableInterface : public detail::TableBase {
public:

    typedef RecordT Record;
    typedef TreeView<Record> Tree;
    typedef boost::transform_iterator<detail::RecordConverter<RecordT>,detail::IteratorBase> Iterator;
    typedef Iterator iterator;
    typedef Iterator const_iterator;

    Tree asTree(TreeMode mode) const { return Tree(*this, mode); }

    Iterator begin() const {
        return Iterator(this->_begin(), detail::RecordConverter<RecordT>());
    }

    Iterator end() const {
        return Iterator(this->_end(), detail::RecordConverter<RecordT>());
    }

    Iterator unlink(Iterator const & iter) const {
        return Iterator(this->_unlink(iter.base()), detail::RecordConverter<RecordT>());
    }

    Iterator find(RecordId id) const {
        return Iterator(this->_find(id), detail::RecordConverter<RecordT>());
    }

    Record operator[](RecordId id) const {
        return detail::Access::makeRecord<RecordT>(this->_get(id));
    }

protected:

    TableInterface(
        Layout const & layout,
        int defaultBlockRecordCount,
        int capacity,
        IdFactory::Ptr const & idFactory = IdFactory::Ptr(),
        AuxBase::Ptr const & aux = AuxBase::Ptr()
    ) : detail::TableBase(layout, defaultBlockRecordCount, capacity, idFactory, aux) {}

    Record _addRecord(RecordId id, AuxBase::Ptr const & aux = AuxBase::Ptr()) const {
        return detail::Access::makeRecord<Record>(this->detail::TableBase::_addRecord(id, aux));
    }

    Record _addRecord(AuxBase::Ptr const & aux = AuxBase::Ptr()) const {
        return detail::Access::makeRecord<Record>(this->detail::TableBase::_addRecord(aux));
    }
    
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_TableInterface_h_INCLUDED
