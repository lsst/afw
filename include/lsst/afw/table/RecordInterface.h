// -*- lsst-c++ -*-
#ifndef AFW_TABLE_RecordInterface_h_INCLUDED
#define AFW_TABLE_RecordInterface_h_INCLUDED

#include "boost/iterator/transform_iterator.hpp"

#include "lsst/base.h"
#include "lsst/afw/table/RecordBase.h"
#include "lsst/afw/table/IteratorBase.h"
#include "lsst/afw/table/TableBase.h"
#include "lsst/afw/table/SchemaMapper.h"
#include "lsst/afw/table/detail/Access.h"


namespace lsst { namespace afw { namespace table {

template <typename RecordT>
struct MappedRecordProxy {
    RecordT record;
    SchemaMapper mapper;
};

template <typename Tag>
class ChildView : private RecordBase {
public:

    typedef typename Tag::Record Record;
    typedef boost::transform_iterator<detail::RecordConverter<Record>,ChildIteratorBase> Iterator;

    Iterator begin() const {
        return Iterator(this->_beginChildren(), detail::RecordConverter<Record>());
    }

    Iterator end() const {
        return Iterator(this->_endChildren(), detail::RecordConverter<Record>());
    }

    explicit ChildView(RecordBase const & record) : RecordBase(record) {}

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
    typedef ChildView<Tag> Children;

    /// @copydoc RecordBase::_getParent
    Record getParent() const {
        return detail::Access::makeRecord<Record>(this->_getParent());
    }

    /**
     *  @brief Return an iterator over the record's children.
     *
     *  @throw LogicErrorException if !getSchema().hasTree().
     */
    Children getChildren() const { return Children(*this); }

    /**
     *  @brief Deep assignment.
     *
     *  The record ID is not copied.
     *
     *  @throw lsst::pex::exceptions::LogicErrorException if other.getSchema() != this->getSchema().
     *
     *  Syntax is intended to mimic afw::Image::operator<<=.
     *
     *  The fact that this is a const member function is admittedly weird, but it makes sense
     *  if you think about it (see the docs for RecordBase).
     */
    Record const & operator<<=(Record const & other) const {
        if (other != *this)
            this->_copyFrom(other);
        return static_cast<Record const &>(*this);
    }

    /**
     *  @brief Deep assignment through a SchemaMapper.
     *
     *  The MappedRecordProxy is a bit of syntactic sugar that allows assignment through
     *  a SchemaMapper to be written as:
     *  @code
     *  outputRecord <<= mapper << inputRecord;
     *  @endcode
     *
     *  The fact that this is a const member function is admittedly weird, but it makes sense
     *  if you think about it (see the docs for RecordBase).
     */
    Record const & operator<<=(MappedRecordProxy<Record> const & other) const;

protected:

    template <typename OtherTag> friend class TableInterface;

    explicit RecordInterface(RecordBase const & other) : RecordBase(other) {}

    RecordInterface(RecordInterface const & other) : RecordBase(other) {}

    void operator=(RecordInterface const & other) { RecordBase::operator=(other); }

};

template <typename Tag>
inline MappedRecordProxy<typename Tag::Record>
operator<<(SchemaMapper const & mapper, RecordInterface<Tag> const & record) {
    MappedRecordProxy<typename Tag::Record> proxy = { mapper, record };
    return proxy;
}

template <typename Tag>
inline typename Tag::Record const &
RecordInterface<Tag>::operator<<=(MappedRecordProxy<typename Tag::Record> const & proxy) const {
    this->_copyFrom(proxy.record, proxy.mapper);
    return static_cast<Record const &>(*this);
}

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_RecordInterface_h_INCLUDED
