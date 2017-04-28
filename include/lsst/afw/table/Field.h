// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Field_h_INCLUDED
#define AFW_TABLE_Field_h_INCLUDED

#include <iostream>

#include "lsst/afw/table/FieldBase.h"

namespace lsst {
namespace afw {
namespace table {

/**
 *  A description of a field in a table.
 *
 *  Field combines a type with the field name, documentation, units,
 *  and in some cases, the size of the field.
 *
 *  Specializations for different field types are inherited through
 *  FieldBase; see the documentation for those specializations for
 *  additional information about particular field types.
 */
template <typename T>
struct Field : public FieldBase<T> {
    /// Type used to store field data in the table (a field may have multiple elements).
    typedef typename FieldBase<T>::Element Element;

    /**
     *  Construct a new field.
     *
     *  @param[in]  name         Name of the field.  Schemas provide extra functionality for names
     *                           whose components are separated by periods.  It may also be practical
     *                           to limit field names to lowercase letters, numbers, and periods,
     *                           as only those names can be round-tripped with FITS I/O (periods are
     *                           converted to underscores in FITS, but hence cannot be distinguished
     *                           from underscores in field names).
     *  @param[in]  doc          Documentation for the field.  Should not contain single-quotes
     *                           to avoid FITS round-trip problems.
     *  @param[in]  units        Units for the field.  Should not contain single-quotes
     *                           to avoid FITS round-trip problems.
     *  @param[in]  size         Size of the field as an integer, if appropriate.  Field types that
     *                           accept a size have a FieldBase that is implicitly constructable from
     *                           an integer, so the argument type should be considered to effectively
     *                           be int; using FieldBase here allows use to throw when the signature
     *                           does not match the field type.
     */
    Field(std::string const& name, std::string const& doc, std::string const& units = "",
          FieldBase<T> const& size = FieldBase<T>())
            : FieldBase<T>(size), _name(name), _doc(doc), _units(units) {}

    /**
     *  Construct a new field.
     *
     *  @param[in]  name         Name of the field.  Schemas provide extra functionality for names
     *                           whose components are separated by periods.  It may also be practical
     *                           to limit field names to lowercase letters, numbers, and periods,
     *                           as only those names can be round-tripped with FITS I/O (periods are
     *                           converted to underscores in FITS, but hence cannot be distinguished
     *                           from underscores in field names).
     *  @param[in]  doc          Documentation for the field.
     *  @param[in]  size         Size of the field as an integer, if appropriate.  Field types that
     *                           accept a size have a FieldBase that is implicitly constructable from
     *                           an integer, so the argument type should be considered to effectively
     *                           be int; using FieldBase here allows use to throw when the signature
     *                           does not match the field type.
     */
    Field(std::string const& name, std::string const& doc, FieldBase<T> const& size)
            : FieldBase<T>(size), _name(name), _doc(doc), _units() {}

    /// Return the name of the field.
    std::string const& getName() const { return _name; }

    /// Return the documentation for the field.
    std::string const& getDoc() const { return _doc; }

    /// Return the units for the field.
    std::string const& getUnits() const { return _units; }

    /// Stringification.
    inline friend std::ostream& operator<<(std::ostream& os, Field<T> const& field) {
        os << "Field['" << Field<T>::getTypeString() << "'](name=\"" << field.getName() << "\"";
        if (!field.getDoc().empty()) os << ", doc=\"" << field.getDoc() << "\"";
        if (!field.getUnits().empty()) os << ", units=\"" << field.getUnits() << "\"";
        field.stream(os);
        return os << ")";
    }

    /// Return a new Field with a new name and other properties the same as this.
    Field<T> copyRenamed(std::string const& newName) const {
        return Field<T>(newName, getDoc(), getUnits(), *this);
    }

private:
    std::string _name;
    std::string _doc;
    std::string _units;
};
}
}
}  // namespace lsst::afw::table

#endif  // !AFW_TABLE_Field_h_INCLUDED
