// -*- lsst-c++ -*-
#ifndef AFW_TABLE_RecordBase_h_INCLUDED
#define AFW_TABLE_RecordBase_h_INCLUDED

#include "lsst/base.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/TableBase.h"

namespace lsst { namespace afw { namespace table {

class SchemaMapper;

class RecordBase : private boost::noncopyable {
public:

    /// @brief Return the Schema that holds this record's fields and keys.
    Schema const & getSchema() const { return _table->getSchema(); }

    /// @brief Return the table this record belongs to.
    CONST_PTR(TableBase) getTable() const { return _table; }

    /**
     *  @brief Return a pointer to the underlying elements of a field (non-const).
     *
     *  This low-level access is intended mostly for use with serialization;
     *  users should generally prefer the safer get(), set() and operator[]
     *  member functions.
     */
    template <typename T>
    typename Field<T>::Element * getElement(Key<T> const & key) {
        return reinterpret_cast<typename Field<T>::Element*>(
            reinterpret_cast<char*>(_data) + key.getOffset()
        );
    }

    /**
     *  @brief Return a pointer to the underlying elements of a field (const).
     *
     *  This low-level access is intended mostly for use with serialization;
     *  users should generally prefer the safer get(), set() and operator[]
     *  member functions.
     */
    template <typename T>
    typename Field<T>::Element const * getElement(Key<T> const & key) const {
        return reinterpret_cast<typename Field<T>::Element const *>(
            reinterpret_cast<char const *>(_data) + key.getOffset()
        );
    }

    /**
     *  @brief Return a reference (or reference-like type) to the field's value.
     *
     *  No checking is done to ensure the Key belongs to the correct schema.
     */
    template <typename T> 
    typename Field<T>::Reference operator[](Key<T> const & key) {
        return key.getReference(getElement(key), _manager);
    }

    /**
     *  @brief Return a const reference (or const-reference-like type) to the field's value.
     *
     *  No checking is done to ensure the Key belongs to the correct schema.
     */
    template <typename T> 
    typename Field<T>::ConstReference operator[](Key<T> const & key) const {
        return key.getConstReference(getElement(key), _manager);
    }
    
    /**
     *  @brief Return the value of a field for the given key.
     *
     *  No checking is done to ensure the Key belongs to the correct schema.
     */
    template <typename T>
    typename Field<T>::Value get(Key<T> const & key) const {
        return key.getValue(getElement(key), _manager);
    }

    /**
     *  @brief Set value of a field for the given key.
     *
     *  This method has an additional template parameter because some fields 
     *  accept and convert different types to the stored field type.
     *
     *  No checking is done to ensure the Key belongs to the correct schema.
     */
    template <typename T, typename U>
    void set(Key<T> const & key, U const & value) {
        key.setValue(getElement(key), _manager, value);
    }

protected:

    /// @brief Construct a record with uninitialized data.
    RecordBase(CONST_PTR(TableBase) const & table) : _table(table) { _table->_initialize(*this); }

    /// @brief Construct a record as a deep copy of another with an identical schema.
    RecordBase(CONST_PTR(TableBase) const & table, RecordBase const & other) :
        _table(table)
    {
        _table->_initialize(*this);
        _assign(other);
    }

    /// @brief Construct a record as a deep copy of another with fields mapped between different schemas.
    RecordBase(CONST_PTR(TableBase) const & table, RecordBase const & other, SchemaMapper const & mapper) :
        _table(table)
    {
        _table->_initialize(*this);
        _assign(other, mapper);
    }

    /// @brief Copy all field values from other to this, requiring that they have equal schemas.
    void _assign(RecordBase const & other);

    /// @brief Copy field values from other to this, using a mapper.
    void _assign(RecordBase const & other, SchemaMapper const & mapper);

    /**
     *  @brief Polymorphic deep-copy with identical schemas.
     *
     *  Public access to this is not provided because some RecordBase subclasses may want to restrict
     *  the table classes they can be associated with.  Record container classes (Vector, Set) use
     *  this implementation.
     *
     *  Callers must guarantee that table->getSchema() == this->getSchema().
     */
    virtual PTR(RecordBase) _clone(CONST_PTR(TableBase) const & table) const = 0;

    /**
     *  @brief Polymorphic deep-copy with a schema mapper.
     *
     *  Public access to this is not provided because some RecordBase subclasses may want to restrict
     *  the table classes they can be associated with.  Record container classes (Vector, Set) use
     *  this implementation.
     *
     *  Callers must guarantee that table->getSchema() == mapper.getOutputSchema() and 
     *  this->getSchema() == mapper.getInputSchema().
     */
    virtual PTR(RecordBase) _clone(CONST_PTR(TableBase) const & table, SchemaMapper const & mapper) const = 0;

private:

    friend class TableBase;

    void * _data;
    CONST_PTR(TableBase) _table;
    ndarray::Manager::Ptr _manager;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_RecordBase_h_INCLUDED
