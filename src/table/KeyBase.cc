// -*- lsst-c++ -*-

#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/afw/table/KeyBase.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/Flag.h"
#include "lsst/afw/table/BaseRecord.h"

namespace lsst { namespace afw { namespace table {

Key<FieldBase<Flag>::Element> KeyBase<Flag>::getStorage() const {
    return detail::Access::extractElement(*this, 0);
}

char const * KeyBase< Coord >::subfields[] = { "ra", "dec" };

template <typename U>
char const * KeyBase< Point<U> >::subfields[] = { "x", "y" };

template <typename U>
char const * KeyBase< Moments<U> >::subfields[] = { "xx", "yy", "xy" };

Key<Angle> KeyBase< Coord >::getRa() const { return detail::Access::extractElement(*this, 0); }

Key<Angle> KeyBase< Coord >::getDec() const { return detail::Access::extractElement(*this, 1); }

template <typename U>
Key<U> KeyBase< Point<U> >::getX() const { return detail::Access::extractElement(*this, 0); }

template <typename U>
Key<U> KeyBase< Point<U> >::getY() const { return detail::Access::extractElement(*this, 1); }

template <typename U>
Key<U> KeyBase< Moments<U> >::getIxx() const { return detail::Access::extractElement(*this, 0); }

template <typename U>
Key<U> KeyBase< Moments<U> >::getIyy() const { return detail::Access::extractElement(*this, 1); }

template <typename U>
Key<U> KeyBase< Moments<U> >::getIxy() const { return detail::Access::extractElement(*this, 2); }

template <typename U>
std::vector<U> KeyBase< Array<U> >::extractVector(BaseRecord const & record) const {
    Key< Array<U> > const * self = static_cast<Key< Array<U> > const *>(this);
    std::vector<U> result(self->getSize());
    typename Key< Array<U> >::ConstReference array = record[*self];
    std::copy(array.begin(), array.end(), result.begin());
    return result;
}

template <typename U>
void KeyBase< Array<U> >::assignVector(BaseRecord & record, std::vector<U> const & values) const {
    Key< Array<U> > const * self = static_cast<Key< Array<U> > const *>(this);
    std::copy(values.begin(), values.end(), record[*self].begin());
}

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
Key<U> KeyBase< Covariance< Moments<U> > >::operator()(int i, int j) const {
    Key< Covariance< Moments<U> > > const * self = static_cast<Key< Covariance< Moments<U> > > const *>(this);
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
