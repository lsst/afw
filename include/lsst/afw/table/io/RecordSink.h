// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_RecordSink_h_INCLUDED
#define AFW_TABLE_IO_RecordSink_h_INCLUDED

#include "lsst/base.h"

namespace lsst { namespace afw { namespace table {

class RecordBase;

namespace io {

/**
 *  @brief A type-erasure base class used to stuff records into any kind of container.
 *
 *  To use it, simply construct a RecordSinkT templated over the container type,
 *  and pass it where a RecordSink is expected.
 *
 *  Sources and sinks are intended as single-pass temporary objects that can easily be
 *  invalidated by mucking with the container they reference.
 */
class RecordSink {
public:

    /// @brief Add a record to the type-erased container.
    virtual void operator()(PTR(RecordBase) const & record) = 0;

    virtual ~RecordSink() {}
};

namespace detail {

template <typename ContainerT>
class RecordSinkT : public RecordSink {
public:

    virtual void operator()(PTR(RecordBase) const & record) {
        _container->insert(_container->end(), record);
    }

    RecordSinkT(ContainerT & container) : _container(&container) {}

private:
    ContainerT * _container;
};

}}}}} // namespace lsst::afw::table::io::detail

#endif // !AFW_TABLE_IO_RecordSink_h_INCLUDED
