// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Schema_h_INCLUDED
#define AFW_TABLE_Schema_h_INCLUDED

#include <set>

#include "boost/shared_ptr.hpp"
#include "boost/ref.hpp"

#include "lsst/ndarray.h"
#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/Field.h"
#include "lsst/afw/table/detail/SchemaData.h"
#include "lsst/afw/table/Flag.h"

namespace lsst { namespace afw { namespace table {

/**
 *  @brief Defines the fields and offsets for a table.
 *
 *  Schema behaves like a container of SchemaItem objects, mapping a descriptive Field object
 *  with the Key object used to access record and ColumnView values.  A Schema is the most
 *  important ingredient in creating a table.
 *
 *  Because offsets for fields are assigned when the field is added to the Schema, 
 *  Schemas do not support removing fields.
 *
 *  A SchemaMapper object can be used to define a relationship between two Schemas to be used
 *  when copying values from one table to another or loading/saving selected fields to disk.
 *
 *  Schema uses copy-on-write, and hence should always be held by value rather than smart pointer.
 *  When creating a Python interface, functions that return Schema by const reference should be
 *  converted to return by value to ensure proper memory management and encapsulation.
 */
class Schema {
    typedef detail::SchemaData Data;
public:

    /// @brief Set type returned by describe().
    typedef std::set<FieldDescription> Description;

    /// @brief Find a SchemaItem in the Schema by name.
    template <typename T>
    SchemaItem<T> find(std::string const & name) const;

    /// @brief Find a SchemaItem in the Schema by key.
    template <typename T>
    SchemaItem<T> find(Key<T> const & key) const;

    /**
     *  @brief Return a set with descriptions of all the fields.
     *
     *  The set will be ordered by field name, not by Key.
     */
    Description describe() const;

    /// @brief Return the raw size of a record in bytes.
    int getRecordSize() const { return _data->_recordSize; }

    /**
     *  @brief Add a new field to the Schema, and return the associated Key.
     *
     *  The offsets of fields are determined by the order they are added, but
     *  may be not contiguous (the Schema may add padding to align fields, and how
     *  much padding is considered an implementation detail).
     */
    template <typename T>
    Key<T> addField(Field<T> const & field);    

    /**
     *  @brief Add a new field to the Schema, and return the associated Key.
     *
     *  This is simply a convenience wrapper, equivalent to:
     *  @code
     *  addField(Field<T>(name, doc, units, base))
     *  @endcode
     */
    template <typename T>
    Key<T> addField(
        std::string const & name, std::string const & doc, std::string const & units = "",
        FieldBase<T> const & base = FieldBase<T>()
    ) {
        return addField(Field<T>(name, doc, units, base));
    }

    /**
     *  @brief Add a new field to the Schema, and return the associated Key.
     *
     *  This is simply a convenience wrapper, equivalent to:
     *  @code
     *  addField(Field<T>(name, doc, base))
     *  @endcode
     */
    template <typename T>
    Key<T> addField(std::string const & name, std::string const & doc, FieldBase<T> const & base) {
        return addField(Field<T>(name, doc, base));
    }

    /// @brief Replace the Field (name/description) for an existing Key.
    template <typename T>
    void replaceField(Key<T> const & key, Field<T> const & field);

    /**
     *  @brief Apply a functor to each SchemaItem in the Schema.
     *
     *  The functor must have a templated or sufficiently overloaded operator() that supports
     *  SchemaItems of all supported field types - even those that are not present in this
     *  particular Schema.
     *
     *  The functor will be passed by value by default; use boost::ref to pass it by reference.
     */
    template <typename F>
    void forEach(F func) const {
        Data::VisitorWrapper<typename boost::unwrap_reference<F>::type &> visitor(func);
        std::for_each(_data->_items.begin(), _data->_items.end(), visitor);
    }

    /// @brief Construct an empty Schema.
    Schema();

private:

    friend class detail::Access;
    
    /// @brief Copy on write; should be called by all mutators.
    void _edit();

    template <typename T>
    Key<T> _addField(Field<T> const & field);    

    Key<Flag> _addField(Field<Flag> const & field);

    boost::shared_ptr<Data> _data;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Schema_h_INCLUDED
