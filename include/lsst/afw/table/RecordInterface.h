// -*- lsst-c++ -*-
#ifndef AFW_TABLE_RecordInterface_h_INCLUDED
#define AFW_TABLE_RecordInterface_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "lsst/base.h"
#include "lsst/afw/table/detail/RecordBase.h"
#include "lsst/afw/table/detail/TreeIteratorBase.h"
#include "lsst/afw/table/detail/IteratorBase.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

template <typename RecordT>
class ChildrenView : private detail::RecordBase {
public:

    typedef boost::transform_iterator<detail::RecordConverter<RecordT>,detail::TreeIteratorBase> Iterator;    

    Iterator begin() const {
        return Iterator(this->_beginChildren(_mode), detail::RecordConverter<RecordT>());
    }

    Iterator end() const {
        return Iterator(this->_endChildren(_mode), detail::RecordConverter<RecordT>()); 
    }

    Iterator find(RecordId id) const {
        return Iterator(this->_find(id), detail::RecordConverter<RecordT>());
    }

    bool empty() const { return this->hasChildren(); }

private:

    template <typename OtherRecordT> friend class RecordInterface;

    ChildrenView(detail::RecordBase const & record, TreeMode mode) : 
        detail::RecordBase(record), _mode(mode) {}

    TreeMode _mode;
};

template <typename RecordT>
class RecordInterface : public detail::RecordBase {
public:

    typedef RecordT Record;
    typedef boost::transform_iterator<detail::RecordConverter<Record>,detail::TreeIteratorBase> TreeIterator;
    typedef boost::transform_iterator<detail::RecordConverter<Record>,detail::IteratorBase> Iterator;
    typedef ChildrenView<Record> Children;

    Record getParent() const {
        return detail::Access::makeRecord<Record>(this->_getParent());
    }

    Children getChildren(TreeMode mode) const {
        return Children(*this, mode);
    }

    TreeIterator asTreeIterator(TreeMode mode) const {
        return TreeIterator(this->_asTreeIterator(mode), detail::RecordConverter<Record>());
    }

    Iterator asIterator() const {
        return Iterator(this->_asIterator(), detail::RecordConverter<Record>());
    }

protected:

    template <typename OtherRecordT> friend class TableInterface;

    Record _addChild(AuxBase::Ptr const & aux = AuxBase::Ptr()) const {
        return detail::Access::makeRecord<Record>(this->detail::RecordBase::_addChild(aux));
    }

    Record _addChild(RecordId id, AuxBase::Ptr const & aux = AuxBase::Ptr()) const {
        return detail::Access::makeRecord<Record>(this->detail::RecordBase::_addChild(id, aux));
    }

    explicit RecordInterface(detail::RecordBase const & other) : detail::RecordBase(other) {}

    RecordInterface(RecordInterface const & other) : detail::RecordBase(other) {}

    void operator=(RecordInterface const & other) { detail::RecordBase::operator=(other); }

};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_RecordInterface_h_INCLUDED
