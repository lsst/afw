#include <limits>

#include "boost/format.hpp"
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/catalog/Field.h"

namespace lsst { namespace catalog {

namespace {

template <typename T> struct TypeTraits;

template <> struct TypeTraits<int> {
    static char const * getName() { return "int"; }
    static int getDefault() { return 0; }
};
template <> struct TypeTraits<float> {
    static char const * getName() { return "float"; }
    static float getDefault() { return std::numeric_limits<float>::quiet_NaN(); }
};
template <> struct TypeTraits<double> {
    static char const * getName() { return "double"; }
    static double getDefault() { return std::numeric_limits<double>::quiet_NaN(); }
};


} // anonyomous

//----- POD scalars -----------------------------------------------------------------------------------------

template <typename T>
std::string Field<T>::getTypeString() const {
    return TypeTraits<T>::getName();
}

template <typename T>
void Field<T>::setDefault(char * buf) const {
    *reinterpret_cast<T*>(buf) = TypeTraits<T>::getDefault();
}

//----- Point scalar ----------------------------------------------------------------------------------------

template <typename U>
std::string Field< Point<U> >::getTypeString() const {
    return (boost::format("Point<%s>") % TypeTraits<U>::getName()).str();
}

template <typename U>
void Field< Point<U> >::setDefault(char * buf) const {
    reinterpret_cast<U*>(buf)[0] = TypeTraits<U>::getDefault();
    reinterpret_cast<U*>(buf)[1] = TypeTraits<U>::getDefault();
}

//----- Shape scalar ----------------------------------------------------------------------------------------

template <typename U>
std::string Field< Shape<U> >::getTypeString() const {
    return (boost::format("Shape<%s>") % TypeTraits<U>::getName()).str();
}

template <typename U>
void Field< Shape<U> >::setDefault(char * buf) const {
    reinterpret_cast<U*>(buf)[0] = TypeTraits<U>::getDefault();
    reinterpret_cast<U*>(buf)[1] = TypeTraits<U>::getDefault();
    reinterpret_cast<U*>(buf)[2] = TypeTraits<U>::getDefault();
}

//----- POD array -------------------------------------------------------------------------------------------

template <typename U>
std::string Field< Array<U> >::getTypeString() const {
    return (boost::format("%s[%d]") % TypeTraits<U>::getName() % this->size).str();
}

template <typename U>
void Field< Array<U> >::setDefault(char * buf) const {
    for (int i = 0; i < size; ++i) {
        reinterpret_cast<U*>(buf)[i] = TypeTraits<U>::getDefault();
    }
}

//----- POD covariance --------------------------------------------------------------------------------------

template <typename U>
std::string Field< Covariance<U> >::getTypeString() const {
    return (boost::format("Cov(%s[%d])") % TypeTraits<U>::getName() % this->size).str();
}

template <typename U>
typename Field< Covariance<U> >::Value Field< Covariance<U> >::getValue(char * buf) const {
    U * p = reinterpret_cast<U*>(buf);
    Value r(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            r(i, j) = p[detail::indexCovariance(i, j)];
        }
    }
    return r;
}

template <typename U>
void Field< Covariance<U> >::setDefault(char * buf) const {
    int const p = detail::computeCovarianceSize(size);
    for (int i = 0; i < p; ++i) {
        reinterpret_cast<U*>(buf)[i] = TypeTraits<U>::getDefault();
    }
}

//----- Point covariance ------------------------------------------------------------------------------------

template <typename U>
std::string Field< Covariance< Point<U> > >::getTypeString() const {
    return (boost::format("Cov(Point<%s>)") % TypeTraits<U>::getName()).str();
}

template <typename U>
typename Field< Covariance< Point<U> > >::Value Field< Covariance< Point<U> > >::getValue(char * buf) const {
    U * p = reinterpret_cast<U*>(buf);
    Value r;
    for (int i = 0; i < r.rows(); ++i) {
        for (int j = 0; j < r.cols(); ++j) {
            r(i, j) = p[detail::indexCovariance(i, j)];
        }
    }
    return r;
}

template <typename U>
void Field< Covariance< Point<U> > >::setDefault(char * buf) const {
    for (int i = 0; i < 3; ++i) {
        reinterpret_cast<U*>(buf)[i] = TypeTraits<U>::getDefault();
    }
}

//----- Shape covariance ------------------------------------------------------------------------------------

template <typename U>
std::string Field< Covariance< Shape<U> > >::getTypeString() const {
    return (boost::format("Cov(Point<%s>)") % TypeTraits<U>::getName()).str();
}

template <typename U>
typename Field< Covariance< Shape<U> > >::Value Field< Covariance< Shape<U> > >::getValue(char * buf) const {
    U * p = reinterpret_cast<U*>(buf);
    Value r;
    for (int i = 0; i < r.rows(); ++i) {
        for (int j = 0; j < r.cols(); ++j) {
            r(i, j) = p[detail::indexCovariance(i, j)];
        }
    }
    return r;
}

template <typename U>
void Field< Covariance< Shape<U> > >::setDefault(char * buf) const {
    for (int i = 0; i < 6; ++i) {
        reinterpret_cast<U*>(buf)[i] = TypeTraits<U>::getDefault();
    }
}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_FIELD(r, data, elem)            \
    template class Field< elem >;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_FIELD, _,
    BOOST_PP_TUPLE_TO_SEQ(CATALOG_FIELD_TYPE_N, CATALOG_FIELD_TYPE_TUPLE)
)

}} // namespace lsst::catalog
