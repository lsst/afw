
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/catalog/detail/KeyAccess.h"

namespace lsst { namespace catalog {

namespace detail {

template <typename T>
Key<typename Field<T>::Element> KeyAccess::extractElement(Key<T> const & key, int offset) {
    typedef typename Field<T>::Element U;
    boost::shared_ptr< KeyData<U> > data = boost::make_shared< KeyData<U> >(Field<U>(key.getField()));
    data->offset = key._data->offset + offset;
    data->nullOffset = key._data->nullOffset;
    data->nullMask = key._data->nullMask;
    return Key<U>(data);
}

} // namespace detail

template <typename U>
Key<U> Key< Point<U> >::getX() const { return detail::KeyAccess::extractElement(*this, 0); }

template <typename U>
Key<U> Key< Point<U> >::getY() const { return detail::KeyAccess::extractElement(*this, 1); }

template <typename U>
Key<U> Key< Shape<U> >::getXX() const { return detail::KeyAccess::extractElement(*this, 0); }

template <typename U>
Key<U> Key< Shape<U> >::getYY() const { return detail::KeyAccess::extractElement(*this, 1); }

template <typename U>
Key<U> Key< Shape<U> >::getXY() const { return detail::KeyAccess::extractElement(*this, 2); }

template <typename U>
Key<U> Key< Array<U> >::at(int i) const {
    if (i >= _data->field.size) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Array key index out of range."
        );
    }
    return detail::KeyAccess::extractElement(*this, i);
}

template <typename U>
Key<U> Key< Covariance<U> >::at(int i, int j) const {
    if (i >= _data->field.size || j >= _data->field.size) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Covariance key index out of range."
        );
    }
    return detail::KeyAccess::extractElement(*this, detail::indexCovariance(i, j));
}

template <typename U>
Key<U> Key< Covariance< Point<U> > >::at(int i, int j) const {
    if (i >= 2 || j >= 2) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Covariance key index out of range."
        );
    }
    return detail::KeyAccess::extractElement(*this, detail::indexCovariance(i, j));
}

template <typename U>
Key<U> Key< Covariance< Shape<U> > >::at(int i, int j) const {
    if (i >= 3 || j >= 3) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Covariance key index out of range."
        );
    }
    return detail::KeyAccess::extractElement(*this, detail::indexCovariance(i, j));
}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_KEY(r, data, elem)            \
    template class Key< elem >;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_KEY, _,
    BOOST_PP_TUPLE_TO_SEQ(CATALOG_FIELD_TYPE_N, CATALOG_FIELD_TYPE_TUPLE)
)

}} // namespace lsst::catalog
