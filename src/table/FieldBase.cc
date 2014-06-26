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
    static char const * getName() { return "I"; }
};
template <> struct TypeTraits<boost::int64_t> {
    static char const * getName() { return "L"; }
};
template <> struct TypeTraits<float> {
    static char const * getName() { return "F"; }
};
template <> struct TypeTraits<double> {
    static char const * getName() { return "D"; }
};
template <> struct TypeTraits<lsst::afw::geom::Angle> {
    static char const * getName() { return "Angle"; }
};
template <> struct TypeTraits<lsst::afw::coord::Coord> {
    static char const * getName() { return "Coord"; }
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
    return (boost::format("Point%s") % TypeTraits<U>::getName()).str();
}

//----- Point scalar ----------------------------------------------------------------------------------------

std::string FieldBase< Coord >::getTypeString() { return "Coord"; }

//----- Moments scalar --------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Moments<U> >::getTypeString() {
    return (boost::format("Moments%s") % TypeTraits<U>::getName()).str();
}

//----- POD array -------------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Array<U> >::getTypeString() {
    return (boost::format("Array%s") % TypeTraits<U>::getName()).str();
}

//----- POD covariance --------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Covariance<U> >::getTypeString() {
    return (boost::format("Cov%s") % TypeTraits<U>::getName()).str();
}

//----- Point covariance ------------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Covariance< Point<U> > >::getTypeString() {
    return (boost::format("CovPoint%s") % TypeTraits<U>::getName()).str();
}

//----- Moments covariance ----------------------------------------------------------------------------------

template <typename U>
std::string FieldBase< Covariance< Moments<U> > >::getTypeString() {
    return (boost::format("CovMoments%s") % TypeTraits<U>::getName()).str();
}

//----- String ----------------------------------------------------------------------------------------------

FieldBase< std::string >::FieldBase(int size) : _size(size) {
    if (size < 0) throw LSST_EXCEPT(
        lsst::pex::exceptions::LengthError,
        "Size must be provided when constructing a string field."
    );
}

std::string FieldBase< std::string >::getTypeString() { return "String"; }

std::string FieldBase< std::string >::getValue(Element const * p, ndarray::Manager::Ptr const & m) const {
    Element const * end = p + _size;
    end = std::find(p, end, 0);
    return std::string(p, end);
}

void FieldBase< std::string >::setValue(
    Element * p, ndarray::Manager::Ptr const &, std::string const & value
) const {
    if (value.size() > std::size_t(_size)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthError,
            (boost::format("String (%d) is too large for field (%d).") % value.size() % _size).str()
        );
    }
    std::copy(value.begin(), value.end(), p);
    std::fill(p + value.size(), p + _size, char(0));
}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_FIELD_BASE(r, data, elem)            \
    template struct FieldBase< elem >;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_FIELD_BASE, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_FIELD_TYPE_N, AFW_TABLE_FIELD_TYPE_TUPLE)
)

}}} // namespace lsst::afw::table
