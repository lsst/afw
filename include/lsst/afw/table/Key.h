// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Key_h_INCLUDED
#define AFW_TABLE_Key_h_INCLUDED

#include "lsst/afw/table/FieldBase.h"
#include "lsst/afw/table/KeyBase.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

class Access;

} // namespace detail

template <typename T>
class Key : public KeyBase<T>, public FieldBase<T> {
public:

    template <typename OtherT> bool operator==(Key<OtherT> const & other) const { return false; }
    template <typename OtherT> bool operator!=(Key<OtherT> const & other) const { return true; }

    bool operator==(Key const & other) const { return _offset == other._offset; }
    bool operator!=(Key const & other) const { return _offset == other._offset; }

    bool operator<(Key const & other) const { return _offset < other._offset; }

private:

    friend class detail::Access;

    explicit Key(int offset, FieldBase<T> const & fb = FieldBase<T>())
        : FieldBase<T>(fb), _offset(offset) {}

    int _offset;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Key_h_INCLUDED
