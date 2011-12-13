// -*- lsst-c++ -*-
#ifndef AFW_TABLE_RecordBase_h_INCLUDED
#define AFW_TABLE_RecordBase_h_INCLUDED

#include "boost/iterator/filter_iterator.hpp"

#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/detail/RecordData.h"
#include "lsst/afw/table/ModificationFlags.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

struct TableImpl;
class ChildFilterPredicate;

} // namespace detail

class SchemaMapper;
class TableBase;
class IteratorBase;

typedef boost::filter_iterator<detail::ChildFilterPredicate,IteratorBase> ChildIteratorBase;


/**
 *  @brief Base class containing most of the implementation for records.
 *
 *  Much of the implementation of derived record classes is provided here
 *  in the form of protected member functions that will need to be wrapped
 *  into public member functions by derived classes.
 *
 *  The all-important field accessors and other member functions that do
 *  not involve the final record type are defined as public member functions.
 *
 *  Final table classes should generally not inherit from RecordBase directly,
 *  and instead should inherit from RecordInterface.
 *
 *  Data is shared between records and tables, but the assertion-based
 *  modification flags are not shared.
 *
 *  @note RecordBase (and RecordBase subclasses) have almost exclusively
 *  const member functions, including mutators.  This reflects the fact that
 *  the underlying data of a record is shared by multiple objects and
 *  record copy-construction is shallow; this means we can trivially
 *  (and accidentally) circumvent const-protection by copy-constructing
 *  a RecordBase from a const reference to a RecordBase.
 *  The only real solution to this problem would be to have a different
 *  class for const records, but that doesn't seem to be worth the trouble
 *  in this case.  The ModificationFlags mechanism, along with the disable()
 *  and makeReadOnly() member functions, provides an assertion-based
 *  substitute for compile-time constness.
 */
class RecordBase : protected ModificationFlags {
public:

    /// @brief Return the Schema that holds this record's fields and keys.
    Schema getSchema() const;

    /// @brief Return true if the record has a parent record.
    bool hasParent() const;

    /// @brief Return true if the record has one or more child records.
    bool hasChildren() const;

    /// @brief Return the unique ID of the record.
    RecordId getId() const { return _data->id; }

    /**
     *  @brief Get the ID of the parent of this record.
     */
    RecordId getParentId() const;

    /**
     *  @brief Set the ID of the parent of this record.
     */
    void setParentId(RecordId id) const;

    /**
     *  @brief Remove the record from whatever table it belongs to.
     *
     *  If the record has already been unlinked (i.e. if !isLinked())
     *  this will always throw LogicErrorException.
     *
     *  If the link mode is POINTERS, records with children cannot
     *  be unlinked (will throw LogicErrorException).
     *
     *  If the link mode is PARENT_ID, records with children may
     *  be removed, but all children must also be removed before
     *  the link mode is set back to POINTERS.
     */
    void unlink() const;

    /// @brief Return true if the record is a member of a table.
    bool isLinked() const { return _data->is_linked(); }

    /**
     *  @brief Return a reference (or reference-like type) that allows the field
     *         to be modified in-place.
     *
     *  Not all fields support reference access, and reference access is only available
     *  when the record's ModificationFlags include the CAN_SET_FIELD bit.  For these
     *  reasons, RecordBase::get should generally be preferred.
     *
     *  No checking is done to ensure the Key belongs to the correct schema.
     */
    template <typename T> 
    typename Field<T>::Reference operator[](Key<T> const & key) const {
        assertBit(CAN_SET_FIELD);
        return detail::Access::getReference(key, _data, _table);
    }
    
    /**
     *  @brief Return the value of a field for the given key.
     *
     *  No checking is done to ensure the Key belongs to the correct schema.
     */
    template <typename T>
    typename Field<T>::Value get(Key<T> const & key) const {
        return detail::Access::getValue(key, _data, _table);
    }

    /**
     *  @brief Set value of a field for the given key.
     *
     *  This method has an additional template parameter because some fields 
     *  accept and convert different types to the correct field type; any
     *  1-d Eigen expression can be used for Array fields, for instance,
     *  not just the exact Eigen::Array class returned by RecordBase::get.
     *
     *  No checking is done to ensure the Key belongs to the correct schema.
     */
    template <typename T, typename U>
    void set(Key<T> const & key, U const & value) const {
        assertBit(CAN_SET_FIELD);
        detail::Access::setValue(key, _data, value, _table);
    }

    /**
     *  @brief Shallow equality comparison.
     *
     *  Returns true only if the records point at the same underlying data.
     */
    bool operator==(RecordBase const & other) const {
        return _data == other._data && _table == other._table;
    }

    /**
     *  @brief Shallow inequality comparison.
     *
     *  Returns false only if the records point at the same underlying data.
     */
    bool operator!=(RecordBase const & other) const {
        return !this->operator==(other);
    }

    /// @brief Disable modifications of the sort defined by the given bit.
    void disable(ModificationFlags::Bit n) { unsetBit(n); }

    /// @brief Disable all modifications.
    void makeReadOnly() { unsetAll(); }

    /**
     *  @brief Shared copy constructor.
     *
     *  All aspects of the record except the modification flags are shared between the two record.
     *  The modification flags will be copied as well, but can then be changed separately.
     */
    RecordBase(RecordBase const & other)
        : ModificationFlags(other), _data(other._data), _table(other._table) {}

    /// Destructor is explicit because class holds a shared_ptr to an incomplete class.
    ~RecordBase();

protected:

    /// @brief Return the record's auxiliary data.
    PTR(AuxBase) getAux() const { return _data->aux; }

    /**
     *  @brief Return the record's parent.
     *
     *  @throw NotFoundException if !hasParent().
     *  @throw LogicErrorException if !getSchema().hasTree().
     */
    RecordBase _getParent() const;

    ChildIteratorBase _beginChildren() const;
    ChildIteratorBase _endChildren() const;

private:

    friend class TableBase;
    friend class IteratorBase;
    friend class SchemaMapper;

    RecordBase() : ModificationFlags(), _data(0), _table() {}

    RecordBase(
        detail::RecordData * data,
        PTR(detail::TableImpl) const & table,
        ModificationFlags const & flags
    ) : ModificationFlags(flags), _data(data), _table(table)
    {}

    detail::RecordData * _data;
    PTR(detail::TableImpl) _table;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_RecordBase_h_INCLUDED
