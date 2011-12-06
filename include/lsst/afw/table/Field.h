// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Field_h_INCLUDED
#define AFW_TABLE_Field_h_INCLUDED


#include "lsst/afw/table/FieldBase.h"
#include "lsst/afw/table/FieldDescription.h"

namespace lsst { namespace afw { namespace table {

template <typename T>
struct Field : public FieldBase<T> {

    Field(
        std::string const & name,
        std::string const & doc,
        FieldBase<T> const & base = FieldBase<T>()
    ) : FieldBase<T>(base), _name(name), _doc(doc) {}

    std::string const & getName() const { return _name; }

    std::string const & getDoc() const { return _doc; }

    FieldDescription describe() const {
        return FieldDescription(getName(), getDoc(), this->getTypeString());
    }

private:
    std::string _name;
    std::string _doc;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Field_h_INCLUDED
