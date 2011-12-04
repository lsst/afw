// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Layout_h_INCLUDED
#define AFW_TABLE_Layout_h_INCLUDED



#include <set>

#include "boost/shared_ptr.hpp"
#include "boost/ref.hpp"

#include "lsst/ndarray.h"
#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/Field.h"
#include "lsst/afw/table//detail/LayoutData.h"

namespace lsst { namespace afw { namespace table {

/**
 *  @brief Defines the fields and offsets for a table.
 *
 *  Layout behaves like a container of LayoutItem objects, mapping a descriptive Field object
 *  with the Key object used to access record and ColumnView values.  A Layout is the most
 *  important ingredient in creating a table.
 *
 *  Because offsets for fields are assigned when the field is added to the Layout, 
 *  Layouts do not support removing fields.
 *
 *  A LayoutMapper object can be used to define a relationship between two Layouts to be used
 *  when copying values from one table to another or loading/saving selected fields to disk.
 *
 *  Layout uses copy-on-write, and hence should always be held by value rather than smart pointer.
 *  When creating a Python interface, functions that return Layout by const reference should be
 *  converted to return by value to ensure proper memory management and encapsulation.
 */
class Layout {
    typedef detail::LayoutData Data;
public:

    /// @brief Set type returned by describe().
    typedef std::set<FieldDescription> Description;

    /// @brief Find a LayoutItem in the Layout by name.
    template <typename T>
    LayoutItem<T> find(std::string const & name) const;

    /// @brief Find a LayoutItem in the Layout by key.
    template <typename T>
    LayoutItem<T> find(Key<T> const & key) const;

    /**
     *  @brief Return a set with descriptions of all the fields.
     *
     *  The set will be ordered by field name, not by Key.
     */
    Description describe() const;

    /// @brief Return the raw size of a record in bytes.
    int getRecordSize() const { return _data->_recordSize; }

    /**
     *  @brief Add a new field to the Layout, and return the associated Key.
     *
     *  The offsets of fields are determined by the order they are added, but
     *  may be not contiguous (the Layout may add padding to align fields, and how
     *  much padding is considered an implementation detail).
     */
    template <typename T>
    Key<T> add(Field<T> const & field);

    /// @brief Replace the Field (name/description) for an existing Key.
    template <typename T>
    void replace(Key<T> const & key, Field<T> const & field);

    /**
     *  @brief Apply a functor to each LayoutItem in the Layout.
     *
     *  The functor must have a templated or sufficiently overloaded operator() that supports
     *  LayoutItems of all supported field types - even those that are not present in this
     *  particular Layout.
     *
     *  The functor will be passed by value by default; use boost::ref to pass it by reference.
     */
    template <typename F>
    void forEach(F func) const {
        Data::VisitorWrapper<typename boost::unwrap_reference<F>::type &> visitor(func);
        std::for_each(_data->_items.begin(), _data->_items.end(), visitor);
    }

    /// @brief Construct an empty Layout.
    Layout();

private:

    friend class detail::Access;
    
    /// @brief Copy on write; should be called by all mutators.
    void _edit();

    boost::shared_ptr<Data> _data;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Layout_h_INCLUDED
