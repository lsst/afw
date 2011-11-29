// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Key_h_INCLUDED
#define AFW_TABLE_Key_h_INCLUDED

#include "lsst/afw/table/config.h"
#include "lsst/afw/table/detail/FieldBase.h"
#include "lsst/afw/table/detail/KeyBase.h"

namespace lsst { namespace afw { namespace table {

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

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Key_h_INCLUDED
