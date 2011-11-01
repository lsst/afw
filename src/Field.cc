#include "boost/format.hpp"

#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/catalog/Field.h"

namespace ndd = lsst::ndarray::detail;

namespace lsst { namespace catalog {

namespace {

template <typename T> struct TypeString;
template <> struct TypeString<int> { static char const * getName() { return "int"; } };
template <> struct TypeString<float> { static char const * getName() { return "float"; } };
template <> struct TypeString<double> { static char const * getName() { return "double"; } };

} // anonyomous

//----- POD scalars -----------------------------------------------------------------------------------------

template <typename T>
std::string Field<T>::getTypeString() const {
    return TypeString<T>::getName();
}

template <typename T>
typename Field<T>::Column Field<T>::makeColumn(
    void * buf, ndarray::DataOrderEnum order, int recordCount, int recordSize,
    ndarray::Manager::Ptr const & manager, NoFieldData const &
) {
    return ndd::ArrayAccess<Column>::construct(
        reinterpret_cast<T*>(buf),
        ndd::Core<1>::create(
            ndarray::makeVector(recordCount),
            ndarray::makeVector((order == ndarray::ROW_MAJOR) ? recordSize : 1),
            manager
        )
    );
}

//----- Point scalar ----------------------------------------------------------------------------------------

template <typename U>
std::string Field< Point<U> >::getTypeString() const {
    return (boost::format("Point<%s>") % TypeString<U>::getName()).str();
}

template <typename U>
typename Field< Point<U> >::Column Field< Point<U> >::makeColumn(
    void * buf, ndarray::DataOrderEnum order, int recordCount, int recordSize,
    ndarray::Manager::Ptr const & manager, NoFieldData const &
) {
    ndd::Core<1>::Ptr core = ndarray::detail::Core<1>::create(
        ndarray::makeVector(recordCount),
        ndarray::makeVector((order == ndarray::ROW_MAJOR) ? recordSize : 2),
        manager
    );
    return Column(
        ndd::ArrayAccess< ndarray::Array<U,1> >::construct(reinterpret_cast<U*>(buf), core),
        ndd::ArrayAccess< ndarray::Array<U,1> >::construct(reinterpret_cast<U*>(buf) + 1, core)
    );
}

//----- Shape scalar ----------------------------------------------------------------------------------------

template <typename U>
std::string Field< Shape<U> >::getTypeString() const {
    return (boost::format("Shape<%s>") % TypeString<U>::getName()).str();
}

template <typename U>
typename Field< Shape<U> >::Column Field< Shape<U> >::makeColumn(
    void * buf, ndarray::DataOrderEnum order, int recordCount, int recordSize,
    ndarray::Manager::Ptr const & manager, NoFieldData const &
) {
    ndd::Core<1>::Ptr core = ndarray::detail::Core<1>::create(
        ndarray::makeVector(recordCount),
        ndarray::makeVector((order == ndarray::ROW_MAJOR) ? recordSize : 3),
        manager
    );
    return Column(
        ndd::ArrayAccess< ndarray::Array<U,1> >::construct(reinterpret_cast<U*>(buf), core),
        ndd::ArrayAccess< ndarray::Array<U,1> >::construct(reinterpret_cast<U*>(buf) + 1, core),
        ndd::ArrayAccess< ndarray::Array<U,1> >::construct(reinterpret_cast<U*>(buf) + 2, core)
    );
}

//----- POD array -------------------------------------------------------------------------------------------

template <typename U>
std::string Field< Array<U> >::getTypeString() const {
    return (boost::format("%s[%d]") % TypeString<U>::getName() % this->size).str();
}

template <typename U>
typename Field< Array<U> >::Column Field< Array<U> >::makeColumn(
    void * buf, ndarray::DataOrderEnum order, int recordCount, int recordSize,
    ndarray::Manager::Ptr const & manager, int size
) {
    return ndarray::detail::ArrayAccess<Column>::construct(
        reinterpret_cast<U*>(buf),
        ndarray::detail::Core<2>::create(
            ndarray::makeVector(recordCount, size),
            ndarray::makeVector((order == ndarray::ROW_MAJOR) ? recordSize : size, 1),
            manager
        )
    );
}

//----- POD covariance --------------------------------------------------------------------------------------

template <typename U>
std::string Field< Covariance<U> >::getTypeString() const {
    return (boost::format("Cov(%s[%d])") % TypeString<U>::getName() % this->size).str();
}

template <typename U>
typename Field< Covariance<U> >::Column Field< Covariance<U> >::makeColumn(
    void * buf, ndarray::DataOrderEnum order, int recordCount, int recordSize,
    ndarray::Manager::Ptr const & manager, int size
) {
    return Column(
        reinterpret_cast<U*>(buf), recordCount, 
        (order == ndarray::ROW_MAJOR) ? recordSize : detail::computePackedSize(size),
        manager, size
    );
}

//----- Point covariance ------------------------------------------------------------------------------------

template <typename U>
std::string Field< Covariance< Point<U> > >::getTypeString() const {
    return (boost::format("Cov(Point<%s>)") % TypeString<U>::getName()).str();
}

template <typename U>
typename Field< Covariance< Point<U> > >::Column Field< Covariance< Point<U> > >::makeColumn(
    void * buf, ndarray::DataOrderEnum order, int recordCount, int recordSize,
    ndarray::Manager::Ptr const & manager, NoFieldData const &
) {
    return Column(
        reinterpret_cast<U*>(buf), recordCount,
        (order == ndarray::ROW_MAJOR) ? recordSize : 3,
        manager
    );
}

//----- Shape covariance ------------------------------------------------------------------------------------

template <typename U>
std::string Field< Covariance< Shape<U> > >::getTypeString() const {
    return (boost::format("Cov(Point<%s>)") % TypeString<U>::getName()).str();
}

template <typename U>
typename Field< Covariance< Shape<U> > >::Column Field< Covariance< Shape<U> > >::makeColumn(
    void * buf, ndarray::DataOrderEnum order, int recordCount, int recordSize,
    ndarray::Manager::Ptr const & manager, NoFieldData const &
) {
    return Column(
        reinterpret_cast<U*>(buf), recordCount, 
        (order == ndarray::ROW_MAJOR) ? recordSize : 6,
        manager
    );
}

#define INSTANTIATE_FIELD(r, data, elem)            \
    template class Field< elem >;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_FIELD, _,
    BOOST_PP_TUPLE_TO_SEQ(CATALOG_FIELD_TYPE_N, CATALOG_FIELD_TYPE_TUPLE)
)

}} // namespace lsst::catalog
