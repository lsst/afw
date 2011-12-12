// -*- lsst-c++ -*-

#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/afw/table/KeyBase.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/Flag.h"

namespace lsst { namespace afw { namespace table {

Key<FieldBase<Flag>::Element> KeyBase<Flag>::getStorage() const {
    return detail::Access::extractElement(*this, 0);
}

template <typename U>
Key<U> KeyBase< Point<U> >::getX() const { return detail::Access::extractElement(*this, 0); }

template <typename U>
Key<U> KeyBase< Point<U> >::getY() const { return detail::Access::extractElement(*this, 1); }

template <typename U>
Key<U> KeyBase< Shape<U> >::getIXX() const { return detail::Access::extractElement(*this, 0); }

template <typename U>
Key<U> KeyBase< Shape<U> >::getIYY() const { return detail::Access::extractElement(*this, 1); }

template <typename U>
Key<U> KeyBase< Shape<U> >::getIXY() const { return detail::Access::extractElement(*this, 2); }

template <typename U>
Key<U> KeyBase< Array<U> >::operator[](int i) const {
    Key< Array<U> > const * self = static_cast<Key< Array<U> > const *>(this);
    if (i >= self->getSize() || i < 0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Array key index out of range."
        );
    }
    return detail::Access::extractElement(*this, i);
}

template <typename U>
Key<U> KeyBase< Covariance<U> >::operator()(int i, int j) const {
    Key< Covariance<U> > const * self = static_cast<Key< Covariance<U> > const *>(this);
    if (i >= self->getSize() || j >= self->getSize() || i < 0 || j < 0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Covariance key index out of range."
        );
    }
    return detail::Access::extractElement(*this, detail::indexCovariance(i, j));
}

template <typename U>
Key<U> KeyBase< Covariance< Point<U> > >::operator()(int i, int j) const {
    Key< Covariance< Point<U> > > const * self = static_cast<Key< Covariance< Point<U> > > const *>(this);
    if (i >= self->getSize() || j >= self->getSize() || i < 0 || j < 0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Covariance key index out of range."
        );
    }
    return detail::Access::extractElement(*this, detail::indexCovariance(i, j));
}

template <typename U>
Key<U> KeyBase< Covariance< Shape<U> > >::operator()(int i, int j) const {
    Key< Covariance< Shape<U> > > const * self = static_cast<Key< Covariance< Shape<U> > > const *>(this);
    if (i >= self->getSize() || j >= self->getSize() || i < 0 || j < 0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Covariance key index out of range."
        );
    }
    return detail::Access::extractElement(*this, detail::indexCovariance(i, j));
}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_KEY(r, data, elem)            \
    template class KeyBase< elem >;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_KEY, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_FIELD_TYPE_N, AFW_TABLE_FIELD_TYPE_TUPLE)
)

}}} // namespace lsst::afw::table
