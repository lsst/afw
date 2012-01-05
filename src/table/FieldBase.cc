// -*- lsst-c++ -*-

#include <limits>

#include "boost/format.hpp"
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/afw/table/FieldBase.h"
#include "lsst/afw/table/Flag.h"

namespace lsst { namespace afw { namespace table {

namespace {

template <typename T> struct TypeTraits;

template <> struct TypeTraits<boost::int32_t> {
    static char const * getName() { return "I4"; }
};
template <> struct TypeTraits<boost::uint32_t> {
    static char const * getName() { return "U4"; }
};
template <> struct TypeTraits<boost::int64_t> {
    static char const * getName() { return "I8"; }
};
template <> struct TypeTraits<boost::uint64_t> {
    static char const * getName() { return "U8"; }
};
template <> struct TypeTraits<float> {
    static char const * getName() { return "F4"; }
};
template <> struct TypeTraits<double> {
    static char const * getName() { return "F8"; }
};

} // anonyomous

//----- POD scalars -----------------------------------------------------------------------------------------

template <typename T>
std::string FieldBase<T>::getTypeString() {
    return TypeTraits<T>::getName();
}

//----- Point scalar ----------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Point<U> >::getTypeString() {
    return (boost::format("Point<%s>") % TypeTraits<U>::getName()).str();
}

//----- Shape scalar ----------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Shape<U> >::getTypeString() {
    return (boost::format("Shape<%s>") % TypeTraits<U>::getName()).str();
}

//----- POD array -------------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Array<U> >::getTypeString() {
    return (boost::format("Array<%s>") % TypeTraits<U>::getName()).str();
}

//----- POD covariance --------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Covariance<U> >::getTypeString() {
    return (boost::format("Cov<%s>") % TypeTraits<U>::getName()).str();
}

//----- Point covariance ------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Covariance< Point<U> > >::getTypeString() {
    return (boost::format("Cov<Point<%s>>") % TypeTraits<U>::getName()).str();
}

//----- Shape covariance ------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Covariance< Shape<U> > >::getTypeString() {
    return (boost::format("Cov<Point<%s>>") % TypeTraits<U>::getName()).str();
}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_FIELD_BASE(r, data, elem)            \
    template class FieldBase< elem >;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_FIELD_BASE, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_FIELD_TYPE_N, AFW_TABLE_FIELD_TYPE_TUPLE)
)

}}} // namespace lsst::afw::table
