// -*- lsst-c++ -*-

#include <cstdint>
#include <limits>

#include "boost/format.hpp"
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/afw/table/FieldBase.h"
#include "lsst/afw/table/Flag.h"

namespace lsst {
namespace afw {
namespace table {

namespace {

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<std::uint8_t> {
    static char const *getName() { return "B"; }
};
template <>
struct TypeTraits<std::uint16_t> {
    static char const *getName() { return "U"; }
};
template <>
struct TypeTraits<std::int32_t> {
    static char const *getName() { return "I"; }
};
template <>
struct TypeTraits<std::int64_t> {
    static char const *getName() { return "L"; }
};
template <>
struct TypeTraits<float> {
    static char const *getName() { return "F"; }
};
template <>
struct TypeTraits<double> {
    static char const *getName() { return "D"; }
};
template <>
struct TypeTraits<lsst::geom::Angle> {
    static char const *getName() { return "Angle"; }
};

}  // anonyomous

//----- POD scalars -----------------------------------------------------------------------------------------

template <typename T>
std::string FieldBase<T>::getTypeString() {
    return TypeTraits<T>::getName();
}

//----- POD array -------------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase<Array<U> >::getTypeString() {
    return (boost::format("Array%s") % TypeTraits<U>::getName()).str();
}

//----- String ----------------------------------------------------------------------------------------------

FieldBase<std::string>::FieldBase(int size) : _size(size) {
    if (size < 0)
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError,
                          "Size must be provided when constructing a string field.");
}

std::string FieldBase<std::string>::getTypeString() { return "String"; }

std::string FieldBase<std::string>::getValue(Element const *p, ndarray::Manager::Ptr const &m) const {
    if (isVariableLength()) {
        // p is a pointer to a std::string; return a copy
        return std::string(*reinterpret_cast<std::string const *>(p));
    } else {
        // p is a char * that is null-terminated only if the string has fewer than _size chars;
        // return a copy as a std::string
        Element const *end = p + _size;
        end = std::find(p, end, 0);
        return std::string(p, end);
    }
}

void FieldBase<std::string>::setValue(Element *p, ndarray::Manager::Ptr const &,
                                      std::string const &value) const {
    if (isVariableLength()) {
        // p is a pointer to a std::string; replace its contents with a copy of `value`
        *reinterpret_cast<std::string *>(p) = value;
    } else {
        // copy the contents of `value` to p through p + _size, null extra characters, if any
        if (value.size() > std::size_t(_size)) {
            throw LSST_EXCEPT(
                    lsst::pex::exceptions::LengthError,
                    (boost::format("String (%d) is too large for field (%d).") % value.size() % _size).str());
        }
        std::copy(value.begin(), value.end(), p);
        std::fill(p + value.size(), p + _size, char(0));  // null extra characters, if any
    }
}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_FIELD_BASE(r, data, elem) template struct FieldBase<elem>;

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_FIELD_BASE, _,
                      BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_FIELD_TYPE_N, AFW_TABLE_FIELD_TYPE_TUPLE))
}
}
}  // namespace lsst::afw::table
