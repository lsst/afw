// -*- c++ -*-
#ifndef CATALOG_Columns_h_INCLUDED
#define CATALOG_Columns_h_INCLUDED

#include "lsst/catalog/Field.h"

#include "ndarray.hpp"

namespace lsst { namespace catalog {

namespace detail {

template <typename T>
struct ColumnTraits {

    typedef ndarray::Array<T,1> Column;

    static Column make(
        Field<T> const & field, char * data, ndarray::Manager::Ptr const & manager,
        int size, int rowStride, int colStride
    ) {
        return ndarray::detail::ArrayAccess< ndarray::Array<T,1> >::construct(
            reinterpret_cast<T*>(data),
            ndarray::detail::Core<1>::create(
                ndarray::makeVector(size), ndarray::makeVector(colStride / sizeof(T)), manager
            )
        );
    }
};

template <typename U>
struct ColumnTraits< Point<U> > {

    typedef Point< ndarray::Array<U,1> > Column;

    static Column make(
        Field<T> const & field, char * data, ndarray::Manager::Ptr const & manager,
        int size, int rowStride, int colStride
    ) {
        return Column(
            ndarray::detail::ArrayAccess< ndarray::Array<U,1> >::construct(
                reinterpret_cast<U*>(data),
                ndarray::detail::Core<1>::create(
                    ndarray::makeVector(size), ndarray::makeVector(colStride / sizeof(U)), manager
                )
            ),
            ndarray::detail::ArrayAccess< ndarray::Array<U,1> >::construct(
                reinterpret_cast<U*>(data + rowStride * sizeof(U)),
                ndarray::detail::Core<1>::create(
                    ndarray::makeVector(size), ndarray::makeVector(colStride / sizeof(U)), manager
                )
            )
        };
    }
};

template <typename U>
struct ColumnTraits< Shape<U> > {

    typedef Shape< ndarray::Array<U,1> > Column;

    static Column make(
        Field<T> const & field, char * data, ndarray::Manager::Ptr const & manager,
        int size, int rowStride, int colStride
    ) {
        return Column(
            ndarray::detail::ArrayAccess< ndarray::Array<U,1> >::construct(
                reinterpret_cast<U*>(data),
                ndarray::detail::Core<1>::create(
                    ndarray::makeVector(size), ndarray::makeVector(colStride / sizeof(U)), manager
                )
            ),
            ndarray::detail::ArrayAccess< ndarray::Array<U,1> >::construct(
                reinterpret_cast<U*>(data + rowStride * sizeof(U)),
                ndarray::detail::Core<1>::create(
                    ndarray::makeVector(size), ndarray::makeVector(colStride / sizeof(U)), manager
                )
            ),
            ndarray::detail::ArrayAccess< ndarray::Array<U,1> >::construct(
                reinterpret_cast<U*>(data + 2 * rowStride * sizeof(U)),
                ndarray::detail::Core<1>::create(
                    ndarray::makeVector(size), ndarray::makeVector(colStride / sizeof(U)), manager
                )
            )
        };
    }
};

template <typename U>
struct ColumnTraits< Vector<U> > {

    typedef ndarray::Array<U,2> Column;

    static Column make(
        Field<T> const & field, char * data, ndarray::Manager::Ptr const & manager,
        int size, int rowStride, int colStride
    ) {
        return ndarray::detail::ArrayAccess< ndarray::Array<U,2> >::construct(
            reinterpret_cast<U*>(data),
            ndarray::detail::Core<2>::create(
                ndarray::makeVector(size, field.size),
                ndarray::makeVector(colStride / sizeof(U), rowStride / sizeof(U)),
                manager
            )
        );
    }
};

template <typename U>
struct ColumnTraits< Covariance<U> > {

};

} // namespace detail
}} // namespace lsst::catalog

#endif // !CATALOG_Columns_h_INCLUDED
