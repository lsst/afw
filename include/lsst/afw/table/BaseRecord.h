// -*- lsst-c++ -*-
#ifndef AFW_TABLE_BaseRecord_h_INCLUDED
#define AFW_TABLE_BaseRecord_h_INCLUDED

#include "lsst/base.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/BaseTable.h"

namespace lsst { namespace afw { namespace table {

class SchemaMapper;
class ColumnView;

class BaseRecord
#ifndef SWIG
    : private daf::base::Citizen,
      private boost::noncopyable
#endif
{
public:

    typedef BaseTable Table;

    /// @brief Return the Schema that holds this record's fields and keys.
    Schema getSchema() const { return _table->getSchema(); }

    /// @brief Return the table this record is associated with.
    CONST_PTR(BaseTable) getTable() const { return _table; }

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

    /// @brief Copy all field values from other to this, requiring that they have equal schemas.
    void assign(BaseRecord const & other);

    /// @brief Copy field values from other to this, using a mapper.
    void assign(BaseRecord const & other, SchemaMapper const & mapper);

    virtual ~BaseRecord() { _table->_destroy(*this); }

protected:

    /// @brief Called by assign() after transferring fields to allow subclass data members to be copied.
    virtual void _assign(BaseRecord const & other) {}

    /// @brief Construct a record with uninitialized data.
    BaseRecord(PTR(BaseTable) const & table) : daf::base::Citizen(typeid(this)), _table(table) {
        table->_initialize(*this);
    }

private:

    friend class BaseTable;
    friend class ColumnView;

    // All these are definitely private, not protected - we don't want derived classes mucking with them.
    void * _data;
    PTR(BaseTable) _table;
    ndarray::Manager::Ptr _manager;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_BaseRecord_h_INCLUDED
