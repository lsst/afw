// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IdFactory_h_INCLUDED
#define AFW_TABLE_IdFactory_h_INCLUDED

#include <memory>

#include "lsst/base.h"
#include "lsst/afw/table/misc.h"

namespace lsst {
namespace afw {
namespace table {

/**
 *  A polymorphic functor base class for generating record IDs for a table.
 *
 *  The IDs produced by an IdFactory need not be sequential, but they must be unique, both with respect
 *  to the IDs it generates itself and those passed to it via the notify() member function.  Valid IDs
 *  must be nonzero, as zero is used to indicate null in some contexts.
 */
class IdFactory {
public:
    /// Return a new unique RecordId.
    virtual RecordId operator()() = 0;

    /// Notify the IdFactory that the given ID has been used and must not be returned by operator().
    virtual void notify(RecordId id) = 0;

    /// Deep-copy the IdFactory.
    virtual std::shared_ptr<IdFactory> clone() const = 0;

    /**
     *  Return a simple IdFactory that simply counts from 1.
     *
     *  This is used when an empty pointer is passed to the BaseTable constructor.
     */
    static std::shared_ptr<IdFactory> makeSimple();

    /**
     *  Return an IdFactory that includes another, fixed ID in the higher-order bits.
     *
     *  @param[in] expId     ID to include in the upper bits via a bitwise OR.
     *  @param[in] reserved  How many bits to reserve for the part of the ID that is unique.
     *
     *  The final record ID will be:
     *
     *      (upper << reserved) | sequence
     */
    static std::shared_ptr<IdFactory> makeSource(RecordId expId, int reserved);

    /**
     * Return the number to pass as the 'reserved' argument to makeSource for
     * an exposure ID with the given maximum number of bits.
     *
     * @param[in]  maxBits  The maximum number of bits an exposure ID can have.
     */
    static int computeReservedFromMaxBits(int maxBits) {
        // Subtract one for signed integers to avoid the sign bit.
        return sizeof(RecordId)*8 - std::is_signed<RecordId>::value - maxBits;
    }

    IdFactory() = default;
    IdFactory(IdFactory const &) = default;
    IdFactory(IdFactory &&) = default;

    // Protected to prevent slicing.
    IdFactory &operator=(IdFactory const &other) = delete;
    IdFactory &operator=(IdFactory &&other) = delete;
    virtual ~IdFactory() = default;

private:
};
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_IdFactory_h_INCLUDED
