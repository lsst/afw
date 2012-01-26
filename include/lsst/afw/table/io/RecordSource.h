// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_RecordSource_h_INCLUDED
#define AFW_TABLE_IO_RecordSource_h_INCLUDED

#include "lsst/base.h"

namespace lsst { namespace afw { namespace table {

class RecordBase;

namespace io {

/**
 *  @brief A type-erasure base class used to get records out of any kind of container.
 *
 *  To use it, simply construct a RecordSourceT templated over the container type,
 *  and pass it where a RecordSource is expected.
 *
 *  Sources and sinks are intended as single-pass temporary objects that can easily be
 *  invalidated by mucking with the container they reference.
 */
class RecordSource {
public:

    /// @brief Return the next record from the type-erased container, or an empty pointer at the end.
    virtual CONST_PTR(RecordBase) operator()() = 0;

    virtual ~RecordSource() {}

};

namespace detail {

template <typename ContainerT>
class RecordSourceT : public RecordSource {
public:

    virtual CONST_PTR(RecordBase) operator()() {
        if (_iter == _end) return CONST_PTR(RecordBase)();
        CONST_PTR(RecordBase) r = _iter;
        ++_iter;
        return r;
    }

    explicit RecordSourceT(ContainerT const & container) : _iter(container.begin()), _end(container.end()) {}

private:
    typename ContainerT::const_iterator _iter;
    typename ContainerT::const_iterator _end;
};

}}}}} // namespace lsst::afw::table::io::detail

#endif // !AFW_TABLE_IO_RecordSource_h_INCLUDED
