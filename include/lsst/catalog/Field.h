// -*- c++ -*-
#ifndef CATALOG_Field_h_INCLUDED
#define CATALOG_Field_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"
#include "lsst/catalog/detail/FieldBase.h"
#include "lsst/catalog/FieldDescription.h"

namespace lsst { namespace catalog {

template <typename T>
struct Field : public detail::FieldBase<T> {

    Field(
        std::string const & name,
        std::string const & doc,
        detail::FieldBase<T> const & base = detail::FieldBase<T>()
    ) : detail::FieldBase<T>(base), _name(name), _doc(doc) {}

    std::string const & getName() const { return _name; }

    std::string const & getDoc() const { return _doc; }

    FieldDescription describe() const {
        return FieldDescription(getName(), getDoc(), this->getTypeString());
    }

private:
    std::string _name;
    std::string _doc;
};

}} // namespace lsst::catalog

#endif // !CATALOG_Field_h_INCLUDED
