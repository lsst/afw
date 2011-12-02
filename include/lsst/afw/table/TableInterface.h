// -*- lsst-c++ -*-
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

template <typename Tag>
class TreeView : private detail::TableBase {
public:

    typedef typename Tag::Table Table;
    typedef typename Tag::Record Record;
    typedef boost::transform_iterator<detail::RecordConverter<Record>,detail::TreeIteratorBase> Iterator;
    typedef Iterator iterator;
    typedef Iterator const_iterator;

    Iterator begin() const {
        return Iterator(this->_beginTree(_mode), detail::RecordConverter<Record>());
    }

    Iterator end() const {
        return Iterator(this->_endTree(_mode), detail::RecordConverter<Record>());
    }

    Iterator unlink(Iterator const & iter) const {
        if (iter.base().getMode() != _mode) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LogicErrorException,
                "TreeView and iterator modes do not agree."
            );
        }
        return Iterator(this->_unlink(iter.base()), detail::RecordConverter<Record>());
    }

private:

    template <typename OtherTag> friend class TableInterface;

    TreeView(detail::TableBase const & table, TreeMode mode) : detail::TableBase(table), _mode(mode) {}

    TreeMode _mode;
};

template <typename Tag>
class TableInterface : public detail::TableBase {
public:

    typedef typename Tag::Table Table;
    typedef typename Tag::Record Record;
    typedef TreeView<Tag> Tree;
    typedef boost::transform_iterator<detail::RecordConverter<Record>,detail::IteratorBase> Iterator;
    typedef Iterator iterator;
    typedef Iterator const_iterator;

    Tree asTree(TreeMode mode) const { return Tree(*this, mode); }

    Iterator begin() const {
        return Iterator(this->_begin(), detail::RecordConverter<Record>());
    }

    Iterator end() const {
        return Iterator(this->_end(), detail::RecordConverter<Record>());
    }

    Iterator unlink(Iterator const & iter) const {
        return Iterator(this->_unlink(iter.base()), detail::RecordConverter<Record>());
    }

    Iterator find(RecordId id) const {
        return Iterator(this->_find(id), detail::RecordConverter<Record>());
    }

    Record operator[](RecordId id) const {
        return detail::Access::makeRecord<Record>(this->_get(id));
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
