#include "lsst/afw/table/detail/fusion_limits.h"

#include <limits>

#include "boost/format.hpp"
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/afw/table/detail/FieldBase.h"

namespace lsst { namespace afw { namespace table { namespace detail {

namespace {

template <typename T> struct TypeTraits;

template <> struct TypeTraits<int> {
    static char const * getName() { return "int"; }
};
template <> struct TypeTraits<float> {
    static char const * getName() { return "float"; }
};
template <> struct TypeTraits<double> {
    static char const * getName() { return "double"; }
};

} // anonyomous

//----- POD scalars -----------------------------------------------------------------------------------------

template <typename T>
std::string FieldBase<T>::getTypeString() const {
    return TypeTraits<T>::getName();
}

//----- Point scalar ----------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Point<U> >::getTypeString() const {
    return (boost::format("Point<%s>") % TypeTraits<U>::getName()).str();
}

//----- Shape scalar ----------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Shape<U> >::getTypeString() const {
    return (boost::format("Shape<%s>") % TypeTraits<U>::getName()).str();
}

//----- POD array -------------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Array<U> >::getTypeString() const {
    return (boost::format("%s[%d]") % TypeTraits<U>::getName() % this->_size).str();
}

//----- POD covariance --------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Covariance<U> >::getTypeString() const {
    return (boost::format("Cov(%s[%d])") % TypeTraits<U>::getName() % this->_size).str();
}

//----- Point covariance ------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Covariance< Point<U> > >::getTypeString() const {
    return (boost::format("Cov(Point<%s>)") % TypeTraits<U>::getName()).str();
}

//----- Shape covariance ------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Covariance< Shape<U> > >::getTypeString() const {
    return (boost::format("Cov(Point<%s>)") % TypeTraits<U>::getName()).str();
}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_FIELD_BASE(r, data, elem)            \
    template class FieldBase< elem >;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_FIELD_BASE, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_FIELD_TYPE_N, AFW_TABLE_FIELD_TYPE_TUPLE)
)

}}}} // namespace lsst::afw::table::detail
