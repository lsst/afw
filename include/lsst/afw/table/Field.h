// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Field_h_INCLUDED
#define AFW_TABLE_Field_h_INCLUDED

#include <iostream>

#include "lsst/afw/table/FieldBase.h"

namespace lsst { namespace afw { namespace table {

/**
 *  @brief A description of a field in a table.
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

    typedef typename FieldBase<T>::Element Element;

    Field(
        std::string const & name,
        std::string const & doc,
        std::string const & units = "",
        FieldBase<T> const & base = FieldBase<T>()
    ) : FieldBase<T>(base), _name(name), _doc(doc), _units(units) {}

    Field(
        std::string const & name,
        std::string const & doc,
        FieldBase<T> const & base
    ) : FieldBase<T>(base), _name(name), _doc(doc), _units() {}

    std::string const & getName() const { return _name; }

    std::string const & getDoc() const { return _doc; }

    std::string const & getUnits() const { return _units; }

    inline friend std::ostream & operator<<(std::ostream & os, Field<T> const & field) {
        os << "Field['" << Field<T>::getTypeString()
           << "'](name=\"" << field.getName() << "\"";
        if (!field.getDoc().empty())
            os << ", doc=\"" << field.getUnits() << "\"";
        if (!field.getUnits().empty())
            os << ", units=\"" << field.getUnits() << "\"";
        field.stream(os);
        return os << ")";
    }

private:
    std::string _name;
    std::string _doc;
    std::string _units;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Field_h_INCLUDED
