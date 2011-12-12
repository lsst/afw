// -*- lsst-c++ -*-
#ifndef AFW_TABLE_ModificationFlags_h_INCLUDED
#define AFW_TABLE_ModificationFlags_h_INCLUDED

#include <bitset>

#include "lsst/pex/exceptions.h"

namespace lsst { namespace afw { namespace table {

/**
 *  @brief A runtime-assertion-based constness substitute.
 *
 *  Records, tables, and iterators will use ModificationFlags to add bits that propagate something
 *  like constness to make up for the fact that we don't really have compile-time constness.  This
 *  should allow us to verify that routines that shouldn't be doing things aren't doing things.
 *
 *  Most objects will use protected inheritance to hold these flags, as this allows this class to
 *  be totally optimized-away when we compile with NDEBUG, while allowing derived classes to
 *  use the assert() member function.
 */
struct ModificationFlags {

    enum Bit {
        CAN_SET_FIELD = 0,  /// Whether we can set field values or obtain references with [] syntax.
        CAN_ADD_RECORD,     /// Whether we can add new records, child or otherwise.
        CAN_UNLINK_RECORD,  /// Whether we can unlink records from tables.
        NUMBER_OF_BITS      /// Number of bits; for internal use only.
    };

    static ModificationFlags const & all();

    /**
     *  @brief Raise a LogicErrorException if the given modification bit is not set.
     *
     *  If NDEBUG is set, this does nothing.  In fact, when NDEBUG is set, the modification bits
     *  are not even propagated.
     */
    void assertBit(Bit n) const {
#ifndef NDEBUG
        if (!_bits[n]) throw LSST_EXCEPT(lsst::pex::exceptions::LogicErrorException, getMessage(n));
#endif
    }

    ModificationFlags & setAll() {
#ifndef NDEBUG
        _bits.set();
#endif
        return *this;
    }

    ModificationFlags & unsetAll() {
#ifndef NDEBUG
        _bits.reset();
#endif
        return *this;
    }

    ModificationFlags & setBit(Bit n) {
#ifndef NDEBUG
        _bits.set(n);
#endif
        return *this;
    }

    ModificationFlags & unsetBit(Bit n) {
#ifndef NDEBUG
        _bits.set(n);
#endif
        return *this;
    }

private:

    static char const * getMessage(Bit n);

#ifndef NDEBUG
    std::bitset<NUMBER_OF_BITS> _bits;
#endif
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_ModificationFlags_h_INCLUDED
