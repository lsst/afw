// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IdFactory_h_INCLUDED
#define AFW_TABLE_IdFactory_h_INCLUDED

#include <memory>

#include "lsst/base.h"
#include "lsst/afw/table/misc.h"

namespace lsst { namespace afw { namespace table {

/**
 *  @brief A polymorphic functor base class for generating record IDs for a table.
 *
 *  The IDs produced by an IdFactory need not be sequential, but they must be unique, both with respect
 *  to the IDs it generates itself and those passed to it via the notify() member function.  Valid IDs
 *  must be nonzero, as zero is used to indicate null in some contexts.
 */
class IdFactory {
public:

    /// @brief Return a new unique RecordId.
    virtual RecordId operator()() = 0;

    /// @brief Notify the IdFactory that the given ID has been used and must not be returned by operator().
    virtual void notify(RecordId id) = 0;

    /// @brief Deep-copy the IdFactory.
    virtual PTR(IdFactory) clone() const = 0;

    /**
     *  @brief Return a simple IdFactory that simply counts from 1.
     *
     *  This is used when an empty pointer is passed to the BaseTable constructor.
     */
    static PTR(IdFactory) makeSimple();

    /**
     *  @brief Return an IdFactory that includes another, fixed ID in the higher-order bits. 
     *
     *  @param[in] expId     ID to include in the upper bits via a bitwise OR.
     *  @param[in] reserved  How many bits to reserve for the part of the ID that is unique.
     *
     *  The final record ID will be:
     *  @code
     *  (upper << reserved) | sequence
     *  @endcode
     */
    static PTR(IdFactory) makeSource(RecordId expId, int reserved);

    virtual ~IdFactory() {}

private:

    // Protected to prevent slicing.
    void operator=(IdFactory const & other) {}

};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_IdFactory_h_INCLUDED
