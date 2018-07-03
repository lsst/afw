// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Schema_h_INCLUDED
#define AFW_TABLE_Schema_h_INCLUDED

#include <memory>
#include <vector>

#include "ndarray.h"
#include "lsst/base.h"
#include "lsst/afw/fits.h"
#include "lsst/afw/fitsDefaults.h"
#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/Field.h"
#include "lsst/afw/table/detail/SchemaImpl.h"
#include "lsst/afw/table/Flag.h"
#include "lsst/afw/table/AliasMap.h"

namespace lsst {
namespace afw {
namespace table {

class SubSchema;
class BaseRecord;

/**
 *  Defines the fields and offsets for a table.
 *
 *  Schema behaves like a container of SchemaItem objects, mapping a descriptive Field object
 *  with the Key object used to access record and ColumnView values.  A Schema is the most
 *  important ingredient in creating a table.
 *
 *  Because offsets for fields are assigned when the field is added to the Schema,
 *  Schemas do not support removing fields, though they do allow renaming.
 *
 *  Field names in Schemas are expected to be underscore-separated names (e.g. 'a_b_c',
 *  but see @ref afwTableFieldNames for the full conventions, including when to use
 *  underscores vs. CamelCase).  The SubSchema
 *  class and Schema::operator[] provide a heirarchical interface to these names, but are
 *  implemented entirely as string splitting/joining operations that ultimately forward to
 *  member functions that operate on the fully-qualified field name, so there is no requirement
 *  that names be separated by underscores, and no performance advantage to using a SubSchema.
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
    // This variable is defined in SchemaImpl, but is replicated here as
    // so that it is available to Python.
    static int const VERSION = detail::SchemaImpl::VERSION;

    /**
     *  Bit flags used when comparing schemas.
     *
     *  All quantities are compared in insertion order, so if two schemas have the same
     *  fields added in opposite order, they will not be considered equal.
     */
    enum ComparisonFlags {
        EQUAL_KEYS = 0x01,     ///< Keys have the same types offsets, and sizes.
        EQUAL_NAMES = 0x02,    ///< Fields have the same names (ordered).
        EQUAL_DOCS = 0x04,     ///< Fields have the same documentation (ordered).
        EQUAL_UNITS = 0x08,    ///< Fields have the same units (ordered).
        EQUAL_FIELDS = 0x0F,   ///< Fields are identical (but aliases may not be).
        EQUAL_ALIASES = 0x10,  ///< Schemas have identical AliasMaps
        IDENTICAL = 0x1F       ///< Everything is the same.
    };

    //@{
    /// Join strings using the field delimiter appropriate for this Schema
    std::string join(std::string const& a, std::string const& b) const;
    std::string join(std::string const& a, std::string const& b, std::string const& c) const {
        return join(join(a, b), c);
    }
    std::string join(std::string const& a, std::string const& b, std::string const& c,
                     std::string const& d) const {
        return join(join(a, b), join(c, d));
    }
    //@}

    /**
     *  Find a SchemaItem in the Schema by name.
     *
     *  Names corresponding to named subfields are accepted, and will
     *  return a SchemaItem whose field is copied from the parent field
     *  with only the name changed.
     */
    template <typename T>
    SchemaItem<T> find(std::string const& name) const;

    /**
     *  Find a SchemaItem in the Schema by key.
     *
     *  Keys corresponding to named subfields are accepted, and will
     *  return a SchemaItem whose field is copied from the parent field
     *  with only the name changed.  Keys corresponding to unnamed
     *  subfields (such as array elements) are not accepted.
     */
    template <typename T>
    SchemaItem<T> find(Key<T> const& key) const;

    /**
     *  Find a SchemaItem by name and run a functor on it.
     *
     *  Names corresponding to named subfields are not accepted.
     *  The given functor must have an overloaded function call
     *  operator that accepts any SchemaItem type (the same as
     *  a functor provided to forEach).
     */
    template <typename F>
    void findAndApply(std::string const& name, F&& func) const {
        _impl->findAndApply(_aliases->apply(name), std::forward<F>(func));
    }

    /**
     *  Look up a (possibly incomplete) name in the Schema.
     *
     *  See SubSchema for more information.
     *
     *  This member function should generally only be used on
     *  "finished" Schemas; modifying a Schema after a SubSchema
     *  to it has been constructed will not allow the proxy to track
     *  the additions, and will invoke the copy-on-write
     *  mechanism of the Schema itself.
     */
    SubSchema operator[](std::string const& name) const;

    /**
     *  Return a set of field names in the schema.
     *
     *  If topOnly==true, return a unique list of only the part
     *  of the names before the first underscore.  For example,
     *  if the full list of field names is ['a_b_c', 'a_d', 'e_f'],
     *  topOnly==true will return ['a', 'e'].
     *
     *  Returns an instance of Python's builtin set in Python.
     *
     *  Aliases are not returned.
     */
    std::set<std::string> getNames(bool topOnly = false) const;

    /// Return the raw size of a record in bytes.
    int getRecordSize() const { return _impl->getRecordSize(); }

    /// The total number of fields.
    int getFieldCount() const { return _impl->getFieldCount(); }

    /// The number of Flag fields.
    int getFlagFieldCount() const { return _impl->getFlagFieldCount(); }

    /// The number of non-Flag fields.
    int getNonFlagFieldCount() const { return _impl->getNonFlagFieldCount(); }

    /**
     *  Add a new field to the Schema, and return the associated Key.
     *
     *  The offsets of fields are determined by the order they are added, but
     *  may be not contiguous (the Schema may add padding to align fields, and how
     *  much padding is considered an implementation detail).
     *
     *  If doReplace is true and the field exists, it will be replaced instead of
     *  throwing an exception.
     */
    template <typename T>
    Key<T> addField(Field<T> const& field, bool doReplace = false);

    /**
     *  Add a new field to the Schema, and return the associated Key.
     *
     *  This is simply a convenience wrapper, equivalent to:
     *
     *      addField(Field<T>(name, doc, units, base), doReplace)
     */
    template <typename T>
    Key<T> addField(std::string const& name, std::string const& doc, std::string const& units = "",
                    FieldBase<T> const& base = FieldBase<T>(), bool doReplace = false) {
        return addField(Field<T>(name, doc, units, base), doReplace);
    }

    /**
     *  Add a new field to the Schema, and return the associated Key.
     *
     *  This is simply a convenience wrapper, equivalent to:
     *
     *      addField(Field<T>(name, doc, base), doReplace)
     */
    template <typename T>
    Key<T> addField(std::string const& name, std::string const& doc, FieldBase<T> const& base,
                    bool doReplace = false) {
        return addField(Field<T>(name, doc, base), doReplace);
    }

    /// Replace the Field (name/description) for an existing Key.
    template <typename T>
    void replaceField(Key<T> const& key, Field<T> const& field);

    /**
     *  Apply a functor to each SchemaItem in the Schema.
     *
     *  The functor must have a templated or sufficiently overloaded operator() that supports
     *  SchemaItems of all supported field types - even those that are not present in this
     *  particular Schema.
     *
     *  Fields will be processed in the order they were added to the schema.
     */
    template <typename F>
    void forEach(F&& func) const {
        Impl::VisitorWrapper<F> visitor(std::forward<F>(func));
        std::for_each(_impl->getItems().begin(), _impl->getItems().end(), visitor);
    }

    //@{
    /**
     *  Equality comparison
     *
     *  Schemas are considered equal according the standard equality operator if their sequence
     *  of keys are identical (same types with the same offsets); names and descriptions of
     *  fields are not considered.  For a more precise comparison, use compare() or contains().
     */
    bool operator==(Schema const& other) const { return compare(other, EQUAL_KEYS); }
    bool operator!=(Schema const& other) const { return !this->operator==(other); }
    //@}

    /**
     *  Do a detailed equality comparison of two schemas.
     *
     *  See ComparisonFlags for a description of the possible return values
     *
     *  @param[in] other   The other schema to compare to.
     *  @param[in] flags   Which types of comparisions to perform.  Flag bits not present here
     *                     will never be returned.
     */
    int compare(Schema const& other, int flags = EQUAL_KEYS) const;

    /**
     *  Test whether the given schema is a subset of this.
     *
     *  This function behaves very similarly to compare(), but ignores fields that are present
     *  in this but absent in other.
     */
    int contains(Schema const& other, int flags = EQUAL_KEYS) const;

    /**
     *  Return true if the given item is in this schema.
     *
     *  The flags must include the EQUAL_KEYS bit, and if the item cannot be found by key no bits
     *  will be set on return.
     */
    template <typename T>
    int contains(SchemaItem<T> const& item, int flags = EQUAL_KEYS) const;

    /**
     *  Return the map of aliases
     *
     *  Note that while this is a const method, it does allow the Schema's aliases to be
     *  edited - this allows the aliases to be modified even after a Table has been constructed
     *  from the Schema.
     *
     *  See AliasMap for more information on schema aliases.
     */
    std::shared_ptr<AliasMap> getAliasMap() const { return _aliases; }

    /**
     *  Set the alias map
     *
     *  This resets the internal pointer to the alias map, disconnecting
     *  this schema from any others it shares aliases with.
     *
     *  Passing a null pointer is equivalent to passing an empty map.
     */
    void setAliasMap(std::shared_ptr<AliasMap> aliases);

    /// Sever the connection between this schema and any others with which it shares aliases
    void disconnectAliases();

    /// Construct an empty Schema.
    Schema();

    /// Copy constructor.
    Schema(Schema const& other);
    Schema(Schema&& other);

    Schema& operator=(Schema const& other);
    Schema& operator=(Schema&& other);
    ~Schema();

    /** Construct from reading a FITS file.
     *
     * Reads from the nominated 'hdu' (0=PHU which cannot be a catalog,
     * afw::fits::DEFAULT_HDU is a special value meaning read from the first HDU with NAXIS != 0).
     */
    static Schema readFits(std::string const& filename, int hdu = fits::DEFAULT_HDU);
    static Schema readFits(fits::MemFileManager& manager, int hdu = fits::DEFAULT_HDU);
    static Schema readFits(fits::Fits& fitsfile);

    /** Construct from reading a FITS header
     *
     * If 'stripMetadata', then the header will be modified,
     * removing the relevant keywords.
     */
    static Schema fromFitsMetadata(daf::base::PropertyList& header, bool stripMetadata = true);

    /// Stringification.
    friend std::ostream& operator<<(std::ostream& os, Schema const& schema);

    /// Get the Citizen corresponding to this Schema (SchemaImpl is what inherits from Citizen).
    daf::base::Citizen& getCitizen() { return *_impl; }

private:
    friend class detail::Access;
    friend class SubSchema;

    /// Copy on write; should be called by all mutators (except for alias mutators).
    void _edit();

    std::shared_ptr<Impl> _impl;
    std::shared_ptr<AliasMap> _aliases;
};

/**
 *  A proxy type for name lookups in a Schema.
 *
 *  Elements of schema names are assumed to be separated by underscores ("a_b_c");
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
 *
 *      Schema schema(false);
 *      Key<int> a_i = schema.addField<int>("a_i", "integer field");
 *      Key< Point<double> > a_p = schema.addField< Point<double> >("a_p", "point field");
 *
 *      assert(schema["a_i"] == a_i);
 *      SubSchema a = schema["a"];
 *      assert(a["i"] == a_i);
 *      Field<int> f_a_i = schema["a_i"];
 *      assert(f_a_i.getDoc() == "integer field");
 *      assert(schema["a_i"] == "a_i");
 *      assert(schema.find("a_p_x") == a_p.getX());
 */
class SubSchema {
    typedef detail::SchemaImpl Impl;

public:
    //@{
    /// Join strings using the field delimiter appropriate for this Schema
    std::string join(std::string const& a, std::string const& b) const;
    std::string join(std::string const& a, std::string const& b, std::string const& c) const {
        return join(join(a, b), c);
    }
    std::string join(std::string const& a, std::string const& b, std::string const& c,
                     std::string const& d) const {
        return join(join(a, b), join(c, d));
    }
    //@}

    /// Find a nested SchemaItem by name.
    template <typename T>
    SchemaItem<T> find(std::string const& name) const;

    /**
     *  Find a nested SchemaItem by name and run a functor on it.
     *
     *  Names corresponding to named subfields are not accepted.
     *  The given functor must have an overloaded function call
     *  operator that accepts any SchemaItem type (the same as
     *  a functor provided to apply or Schema::forEach).
     */
    template <typename F>
    void findAndApply(std::string const& name, F&& func) const {
        _impl->findAndApply(_aliases->apply(join(_name, name)), std::forward<F>(func));
    }

    /**
     *  Run functor on the SchemaItem represented by this SubSchema
     *
     *  The given functor must have an overloaded function call operator that
     *  accepts any SchemaItem type (the same as a functor provided to apply
     *  or Schema::forEach).
     *
     *  @throws Throws pex::exceptions::NotFoundError if the SubSchemas prefix
     *          does not correspond to the full name of a regular field (not a
     *          named subfield).
     */
    template <typename F>
    void apply(F&& func) const {
        _impl->findAndApply(_aliases->apply(_name), std::forward<F>(func));
    }

    /// Return a nested proxy.
    SubSchema operator[](std::string const& name) const;

    /// Return the prefix that defines this SubSchema relative to its parent Schema.
    std::string const& getPrefix() const { return _name; }

    /**
     *  Return a set of nested names that start with the SubSchema's prefix.
     *
     *  Returns an instance of Python's builtin set in Python.
     *  @see Schema::getNames
     */
    std::set<std::string> getNames(bool topOnly = false) const;

    /**
     *  Implicit conversion to the appropriate Key type.
     *
     *  Implicit conversion operators that are invoked via assignment cannot
     *  be translated to Python.  Instead, the Python wrappers provide an
     *  equivalent asKey() method.
     */
    template <typename T>
    operator Key<T>() const {
        return _impl->find<T>(_aliases->apply(_name)).key;
    }

    /**
     *  Implicit conversion to the appropriate Key type.
     *
     *  Implicit conversion operators that are invoked via assignment cannot
     *  be translated to Python.  Instead, the Python wrappers provide an
     *  equivalent asField() method.
     */
    template <typename T>
    operator Field<T>() const {
        return _impl->find<T>(_aliases->apply(_name)).field;
    }

private:
    friend class Schema;

    SubSchema(std::shared_ptr<Impl> impl, std::shared_ptr<AliasMap> aliases, std::string const& name);

    std::shared_ptr<Impl> _impl;
    std::shared_ptr<AliasMap> _aliases;
    std::string _name;
};

inline SubSchema Schema::operator[](std::string const& name) const {
    return SubSchema(_impl, _aliases, name);
}
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_Schema_h_INCLUDED
