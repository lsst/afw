// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Field_h_INCLUDED
#define AFW_TABLE_Field_h_INCLUDED


#include "lsst/afw/table/FieldBase.h"
#include "lsst/afw/table/FieldDescription.h"

namespace lsst { namespace afw { namespace table {

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

    FieldDescription describe() const {
        return FieldDescription(getName(), getDoc(), getUnits(), this->getTypeString());
    }

private:
    std::string _name;
    std::string _doc;
    std::string _units;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Field_h_INCLUDED
