// -*- c++ -*-
#ifndef CATALOG_Key_h_INCLUDED
#define CATALOG_Key_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"
#include "lsst/catalog/detail/FieldBase.h"
#include "lsst/catalog/detail/KeyBase.h"

namespace lsst { namespace catalog {

template <typename T>
class Key : public detail::KeyBase<T>, public detail::FieldBase<T> {
public:

    template <typename OtherT> bool operator==(Key<OtherT> const & other) const { return false; }
    template <typename OtherT> bool operator!=(Key<OtherT> const & other) const { return true; }

    bool operator==(Key const & other) const { return _offset == other._offset; }
    bool operator!=(Key const & other) const { return _offset == other._offset; }

private:

    friend class detail::Access;

    explicit Key(int offset, detail::FieldBase<T> const & fb = detail::FieldBase<T>())
        : detail::FieldBase<T>(fb), _offset(offset) {}

    int _offset;
};

}} // namespace lsst::catalog

#endif // !CATALOG_Key_h_INCLUDED
