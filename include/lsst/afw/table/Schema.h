// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Schema_h_INCLUDED
#define AFW_TABLE_Schema_h_INCLUDED

#include <vector>

#include "boost/shared_ptr.hpp"
#include "boost/ref.hpp"

#include "ndarray.h"
#include "lsst/base.h"
#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/Field.h"
#include "lsst/afw/table/detail/SchemaImpl.h"
#include "lsst/afw/table/Flag.h"
#include "lsst/afw/table/AliasMap.h"

namespace lsst { namespace afw { namespace table {

class SubSchema;
class BaseRecord;

/**
 *  @brief Defines the fields and offsets for a table.
 *
 *  Schema behaves like a container of SchemaItem objects, mapping a descriptive Field object
 *  with the Key object used to access record and ColumnView values.  A Schema is the most
 *  important ingredient in creating a table.
 *
 *  Because offsets for fields are assigned when the field is added to the Schema, 
 *  Schemas do not support removing fields, though they do allow renaming.
 *
 *  Field names in Schemas are expected to be dot-separated names (e.g. 'a.b.c').  The SubSchema
 *  class and Schema::operator[] provide a heirarchical interface to these names, but are
 *  implemented entirely as string splitting/joining operations that ultimately forward to
 *  member functions that operate on the fully-qualified field name, so there is no requirement
 *  that names be separated by periods, and no performance advantage to using a SubSchema.
 *
 *  A SchemaMapper object can be used to define a relationship between two Schemas to be used
 *  when copying values from one table to another or loading/saving selected fields to disk.
 *
 *  Schema uses copy-on-write, and hence should always be held by value rather than smart pointer.
 *  When creating a Python interface, functions that return Schema by const reference should be
 *  converted to return by value (%returnCopy) to ensure proper memory management and encapsulation.
 */
class Schema {
    typedef detail::SchemaImpl Impl;
public:

    /**
     *  @brief Bit flags used when comparing schemas.
     *
     *  All quantities are compared in insertion order, so if two schemas have the same
     *  fields added in opposite order, they will not be considered equal.
     */
    enum ComparisonFlags {
        EQUAL_KEYS         =0x01, ///< Keys have the same types offsets, and sizes.
        EQUAL_NAMES        =0x02, ///< Fields have the same names (ordered).
        EQUAL_DOCS         =0x04, ///< Fields have the same documentation (ordered).
        EQUAL_UNITS        =0x08, ///< Fields have the same units (ordered).
        IDENTICAL          =0x0F  ///< Everything is the same.
    };

    /**
     *  @brief Find a SchemaItem in the Schema by name.
     *
     *  Names corresponding to named subfields are accepted, and will
     *  return a SchemaItem whose field is copied from the parent field
     *  with only the name changed.
     */
    template <typename T>
    SchemaItem<T> find(std::string name) const;

    /**
     *  @brief Find a SchemaItem in the Schema by key.
     *
     *  Keys corresponding to named subfields are accepted, and will
     *  return a SchemaItem whose field is copied from the parent field
     *  with only the name changed.  Keys corresponding to unnamed
     *  subfields (such as array elements) are not accepted.
     */
    template <typename T>
    SchemaItem<T> find(Key<T> const & key) const;

    /**
     *  @brief Look up a (possibly incomplete) name in the Schema.
     *
     *  See SubSchema for more information.
     *
     *  This member function should generally only be used on
     *  "finished" Schemas; modifying a Schema after a SubSchema
     *  to it has been constructed will not allow the proxy to track
     *  the additions, and will invoke the copy-on-write
     *  mechanism of the Schema itself.
     */
    SubSchema operator[](std::string const & name) const;

    /**
     *  @brief Return a set of field names in the schema.
     *
     *  If topOnly==true, return a unique list of only the part
     *  of the names before the first period.  For example,
     *  if the full list of field names is ['a.b.c', 'a.d', 'e.f'],
     *  topOnly==true will return ['a', 'e'].
     *
     *  Returns an instance of Python's builtin set in Python.
     *
     *  Aliases are not returned.
     */
    std::set<std::string> getNames(bool topOnly=false) const;

    /// @brief Return the raw size of a record in bytes.
    int getRecordSize() const { return _impl->getRecordSize(); }

    /// The total number of fields.
    int getFieldCount() const { return _impl->getFieldCount(); }

    /// The number of Flag fields.
    int getFlagFieldCount() const { return _impl->getFlagFieldCount(); }

    /// The number of non-Flag fields.
    int getNonFlagFieldCount() const { return _impl->getNonFlagFieldCount(); }

    /**
     *  @brief Add a new field to the Schema, and return the associated Key.
     *
     *  The offsets of fields are determined by the order they are added, but
     *  may be not contiguous (the Schema may add padding to align fields, and how
     *  much padding is considered an implementation detail).
     *
     *  If doReplace is true and the field exists, it will be replaced instead of
     *  throwing an exception.
     */
    template <typename T>
    Key<T> addField(Field<T> const & field, bool doReplace=false);

    /**
     *  @brief Add a new field to the Schema, and return the associated Key.
     *
     *  This is simply a convenience wrapper, equivalent to:
     *  @code
     *  addField(Field<T>(name, doc, units, base), doReplace)
     *  @endcode
     */
    template <typename T>
    Key<T> addField(
        std::string const & name, std::string const & doc, std::string const & units = "",
        FieldBase<T> const & base = FieldBase<T>(), bool doReplace=false
    ) {
        return addField(Field<T>(name, doc, units, base), doReplace);
    }

    /**
     *  @brief Add a new field to the Schema, and return the associated Key.
     *
     *  This is simply a convenience wrapper, equivalent to:
     *  @code
     *  addField(Field<T>(name, doc, base), doReplace)
     *  @endcode
     */
    template <typename T>
    Key<T> addField(
        std::string const & name, std::string const & doc, FieldBase<T> const & base,
        bool doReplace=false
    ) {
        return addField(Field<T>(name, doc, base), doReplace);
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
     *
     *  Fields will be processed in the order they were added to the schema.
     */
    template <typename F>
    void forEach(F func) const {
        Impl::VisitorWrapper<typename boost::unwrap_reference<F>::type &> visitor(func);
        std::for_each(_impl->getItems().begin(), _impl->getItems().end(), visitor);
    }

    //@{
    /**
     *  @brief Equality comparison
     *
     *  Schemas are considered equal according the standard equality operator if their sequence
     *  of keys are identical (same types with the same offsets); names and descriptions of
     *  fields are not considered.  For a more precise comparison, use compare() or contains().
     */
    bool operator==(Schema const & other) const { return compare(other, EQUAL_KEYS); }
    bool operator!=(Schema const & other) const { return !this->operator==(other); }
    //@}

    /**
     *  @brief Do a detailed equality comparison of two schemas.
     *
     *  See ComparisonFlags for a description of the possible return values
     *
     *  @param[in] other   The other schema to compare to.
     *  @param[in] flags   Which types of comparisions to perform.  Flag bits not present here
     *                     will never be returned.
     */
    int compare(Schema const & other, int flags=EQUAL_KEYS) const;

    /**
     *  @brief Test whether the given schema is a subset of this.
     *
     *  This function behaves very similarly to compare(), but ignores fields that are present
     *  in this but absent in other.
     */
    int contains(Schema const & other, int flags=EQUAL_KEYS) const;

    /**
     *  @brief Return true if the given item is in this schema.
     *
     *  The flags must include the EQUAL_KEYS bit, and if the item cannot be found by key no bits
     *  will be set on return.
     */
    template <typename T>
    int contains(SchemaItem<T> const & item, int flags=EQUAL_KEYS) const;

    /**
     *  Return the map of aliases
     *
     *  Note that while this is a const method, it does allow the Schema's aliases to be
     *  edited - this allows the aliases to be modified even after a Table has been constructed
     *  from the Schema.
     *
     *  See AliasMap for more information on schema aliases.
     */
    PTR(AliasMap) getAliases() const { return _aliases; }

    /**
     *  Set the alias map
     *
     *  This resets the internal pointer to the alias map, disconnecting
     *  this schema from any others it shares aliases with.
     *
     *  Passing a null pointer is equivalent to passing an empty map.
     */
    void setAliases(PTR(AliasMap) aliases);

    /// Sever the connection between this schema and any others with which it shares aliases
    void disconnectAliases();

    /// @brief Construct an empty Schema.
    explicit Schema();

    /// @brief Copy constructor.
    Schema(Schema const & other);

    /**
     *  @brief Construct from a PropertyList, interpreting it as a FITS binary table header.
     *
     *  @param[in,out] metadata       PropertyList that contains the FITS header keys
     *                                corresponding to a binary table extension.  We can't
     *                                use a PropertySet here, because the order does matter.
     *  @param[in]     stripMetadata  If true, the keys used to define the schema will be removed
     *                                from the PropertySet.
     *
     *  If the column types in the FITS header are not compatible with Schema field types,
     *  of if some required keys (TTYPEn, TFORMn) are not present for some columns,
     *  afw::fits::FitsError will be thrown.
     *
     *  This constructor does not support strong exception safety guarantee when stripMetadata is True;
     *  the PropertyList may be modified when an exception is thrown.
     */
    explicit Schema(daf::base::PropertyList & metadata, bool stripMetadata);

    /**
     *  @brief Construct from a PropertyList, interpreting it as a FITS binary table header.
     *
     *  @param[in,out] metadata       PropertyList that contains the FITS header keys
     *                                corresponding to a binary table extension.  We can't
     *                                use a PropertySet here, because the order does matter.
     *
     *  If the column types in the FITS header are not compatible with Schema field types,
     *  of if some required keys (TTYPEn, TFORMn) are not present for some columns,
     *  afw::fits::FitsError will be thrown.
     *
     *  This overload never strips metadata, allowing it to accept a const PropertyList.
     */
    explicit Schema(daf::base::PropertyList const & metadata);

    /// Stringification.
    friend std::ostream & operator<<(std::ostream & os, Schema const & schema);

    /// @brief Get the Citizen corresponding to this Schema (SchemaImpl is what inherits from Citizen).
    daf::base::Citizen & getCitizen() { return *_impl; }

private:

    friend class detail::Access;
    friend class SubSchema;

    /// @brief Copy on write; should be called by all mutators (except for alias mutators).
    void _edit();

    PTR(Impl) _impl;
    PTR(AliasMap) _aliases;
};

/**
 *  @brief A proxy type for name lookups in a Schema.
 *
 *  Elements of schema names are assumed to be separated by periods ("a.b.c.d");
 *  an incomplete lookup is one that does not resolve to a field.  Not that even
 *  complete lookups can have nested names; a Point field, for instance, has "x"
 *  and "y" nested names.
 *
 *  This proxy object is implicitly convertible to both the appropriate Key type
 *  and the appropriate Field type, if the name is a complete one, and supports
 *  additional find() operations for nested names.
 *
 *  SubSchema is implemented as a proxy that essentially calls Schema::find
 *  after concatenating strings.  It does not provide any performance advantage
 *  over using Schema::find directly.  It is also lazy, so looking up a name
 *  prefix that does not exist within the schema is not considered an error
 *  until the proxy is used.
 *
 *  Some examples:
 *  @code
 *  Schema schema(false);
 *  Key<int> a_i = schema.addField<int>("a.i", "integer field");
 *  Key< Point<double> > a_p = schema.addField< Point<double> >("a.p", "point field");
 *  
 *  assert(schema["a.i"] == a_i);
 *  SubSchema a = schema["a"];
 *  assert(a["i"] == a_i);
 *  Field<int> f_a_i = schema["a.i"];
 *  assert(f_a_i.getDoc() == "integer field");
 *  assert(schema["a.i"] == "a.i");
 *  assert(schema.find("a.p.x") == a_p.getX());
 *  @endcode
 */
class SubSchema {
    typedef detail::SchemaImpl Impl;
public:
    
    /// @brief Find a nested SchemaItem by name.
    template <typename T>
    SchemaItem<T> find(std::string const & name) const;

    /// @brief Return a nested proxy.
    SubSchema operator[](std::string const & name) const;

    /// @brief Return the prefix that defines this SubSchema relative to its parent Schema.
    std::string const & getPrefix() const { return _name; }

    /**
     *  @brief Return a set of nested names that start with the SubSchema's prefix.
     *
     *  Returns an instance of Python's builtin set in Python.
     *  @sa Schema::getNames
     */
    std::set<std::string> getNames(bool topOnly=false) const;

    /**
     *  @brief Implicit conversion to the appropriate Key type.
     *
     *  Implicit conversion operators cannot be translated to Python.  Instead, the SWIG
     *  wrappers provide an equivalent asKey() method.
     */
    template <typename T>
    operator Key<T>() const { return _impl->find<T>(_name).key; }

    /**
     *  @brief Implicit conversion to the appropriate Key type.
     *
     *  Implicit conversion operators cannot be translated to Python.  Instead, the SWIG
     *  wrappers provide an equivalent asField() method.
     */
    template <typename T>
    operator Field<T>() const { return _impl->find<T>(_name).field; }

private:

    friend class Schema;

    SubSchema(PTR(Impl) impl, PTR(AliasMap) aliases, std::string const & name);

    PTR(Impl) _impl;
    PTR(AliasMap) _aliases;
    std::string _name;
};

inline SubSchema Schema::operator[](std::string const & name) const {
    return SubSchema(_impl, _aliases, name);
}

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Schema_h_INCLUDED
