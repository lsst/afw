// -*- lsst-c++ -*-
#ifndef AFW_TABLE_RecordInterface_h_INCLUDED
#define AFW_TABLE_RecordInterface_h_INCLUDED

#include "boost/iterator/transform_iterator.hpp"

#include "lsst/base.h"
#include "lsst/afw/table/RecordBase.h"
#include "lsst/afw/table/TreeIteratorBase.h"
#include "lsst/afw/table/IteratorBase.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

/**
 *  @brief A container-like view into the children of a particular record.
 *
 *  The iterators on a ChildrenView are tree iterators, and can either process
 *  one level of children or all recursive children, depending on the TreeMode
 *  the view was constructed with.
 *
 *  @sa RecordInterface::getChildren().
 */
template <typename Tag>
class ChildrenView : private RecordBase {
public:

    typedef typename Tag::Record Record; ///< @brief Record type obtained by dereferencing iterators.

    typedef boost::transform_iterator<detail::RecordConverter<Record>,TreeIteratorBase> Iterator;    

    Iterator begin() const {
        return Iterator(this->_beginChildren(_mode), detail::RecordConverter<Record>());
    }

    Iterator end() const {
        return Iterator(this->_endChildren(_mode), detail::RecordConverter<Record>()); 
    }

    bool empty() const { return this->hasChildren(); }

private:

    template <typename OtherTag> friend class RecordInterface;

    ChildrenView(RecordBase const & record, TreeMode mode) : 
        RecordBase(record), _mode(mode) {}

    TreeMode _mode;
};

/**
 *  @brief A facade base class that provides most of the public interface of a record.
 *
 *  RecordInterface inherits from RecordBase and gives a record a consistent public interface
 *  by providing thin wrappers around RecordBase member functions that return adapted types.
 *
 *  Final record classes should inherit from RecordInterface, templated on a tag class that
 *  typedefs both the record class and the corresponding final record class (see SimpleRecord
 *  for an example).
 *
 *  RecordInterface does not provide public wrappers for member functions that add new records,
 *  because final record classes may want to control what auxiliary data is required to be present
 *  in a record.
 */
template <typename Tag>
class RecordInterface : public RecordBase {
public:

    typedef typename Tag::Record Record;
    typedef boost::transform_iterator<detail::RecordConverter<Record>,TreeIteratorBase> TreeIterator;
    typedef boost::transform_iterator<detail::RecordConverter<Record>,IteratorBase> Iterator;
    typedef ChildrenView<Tag> Children;

    /// @brief Return the record's parent, or throw NotFoundException if !hasParent().
    Record getParent() const {
        return detail::Access::makeRecord<Record>(this->_getParent());
    }

    /// @brief Return a container-like view into the children of this record.
    Children getChildren(TreeMode mode) const {
        return Children(*this, mode);
    }

    //@{
    /**
     *  @brief Return an iterator that points at the record.
     *
     *  Unlike STL containers, records contain the pointers used to implement their iterators,
     *  and hence can be turned into iterators (as long as isLinked() is true).
     *
     *  These are primarily useful as a way to convert between iterators; the most common use
     *  case is to locate a record by ID using TableInterface::operator[] or TableInterface::find, and then
     *  using asTreeIterator to switch to a different iteration order.
     */
    TreeIterator asTreeIterator(TreeMode mode) const {
        return TreeIterator(this->_asTreeIterator(mode), detail::RecordConverter<Record>());
    }
    Iterator asIterator() const {
        return Iterator(this->_asIterator(), detail::RecordConverter<Record>());
    }
    //@}

protected:

    template <typename OtherTag> friend class TableInterface;

    Record _addChild(AuxBase::Ptr const & aux = AuxBase::Ptr()) const {
        return detail::Access::makeRecord<Record>(this->RecordBase::_addChild(aux));
    }

    Record _addChild(RecordId id, AuxBase::Ptr const & aux = AuxBase::Ptr()) const {
        return detail::Access::makeRecord<Record>(this->RecordBase::_addChild(id, aux));
    }

    explicit RecordInterface(RecordBase const & other) : RecordBase(other) {}

    RecordInterface(RecordInterface const & other) : RecordBase(other) {}

    void operator=(RecordInterface const & other) { RecordBase::operator=(other); }

};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_RecordInterface_h_INCLUDED
