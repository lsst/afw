// -*- c++ -*-
#ifndef AFW_TABLE_RecordInterface_h_INCLUDED
#define AFW_TABLE_RecordInterface_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "lsst/base.h"
#include "lsst/afw/table/detail/RecordBase.h"
#include "lsst/afw/table/detail/TreeIteratorBase.h"
#include "lsst/afw/table/detail/SetIteratorBase.h"
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

    template <typename OtherRecordT, typename RecordAuxT> friend class RecordInterface;

    ChildrenView(detail::RecordBase const & record, IteratorMode mode) : 
        detail::RecordBase(record), _mode(mode) {}

    IteratorMode _mode;
};

template <typename RecordT, typename RecordAuxT=AuxBase>
class RecordInterface : public detail::RecordBase {
public:

    typedef RecordT Record;
    typedef boost::transform_iterator<detail::RecordConverter<RecordT>,detail::TreeIteratorBase> TreeIterator;
    typedef boost::transform_iterator<detail::RecordConverter<RecordT>,detail::SetIteratorBase> SetIterator;
    typedef ChildrenView<RecordT> Children;

    Record getParent() const {
        return detail::Access::makeRecord<RecordT>(this->_getParent());
    }

    Children getChildren(IteratorMode mode) const {
        return Children(*this, mode);
    }

    TreeIterator asTreeIterator(IteratorMode mode) const {
        return TreeIterator(this->_asTreeIterator(mode), detail::RecordConverter<RecordT>());
    }

    SetIterator asSetIterator() const {
        return SetIterator(this->_asSetIterator(), detail::RecordConverter<RecordT>());
    }

protected:

    template <typename OtherRecordT, typename TableAuxT> friend class TableInterface;

    typedef RecordAuxT RecordAux;

    PTR(RecordAux) getAux() const {
        return boost::static_pointer_cast<RecordAux>(detail::RecordBase::getAux());
    }

    RecordT _addChild(PTR(RecordAux) const & aux = PTR(RecordAux)()) {
        return detail::Access::makeRecord<RecordT>(this->detail::RecordBase::_addChild(aux));
    }

    RecordT _addChild(RecordId id, PTR(RecordAux) const & aux = PTR(RecordAux)()) {
        return detail::Access::makeRecord<RecordT>(this->detail::RecordBase::_addChild(id, aux));
    }

    explicit RecordInterface(detail::RecordBase const & other) : detail::RecordBase(other) {}

    RecordInterface(RecordInterface const & other) : detail::RecordBase(other) {}

    void operator=(RecordInterface const & other) { detail::RecordBase::operator=(other); }

};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_RecordInterface_h_INCLUDED
