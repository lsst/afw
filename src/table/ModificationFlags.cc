#include "lsst/afw/table/ModificationFlags.h"

namespace lsst { namespace afw { namespace table {

ModificationFlags const & ModificationFlags::all() {
    static ModificationFlags instance = ModificationFlags().setAll();
    return instance;
}

char const * ModificationFlags::getMessage(Bit n) {
    static char const * messages[] = {
        "Field values cannot be set via this record/table/iterator.",
        "New records cannot be added via this record/table/iterator.",
        "Records cannot be unlinked via this record/table/iterator."
    };
    return messages[n];
}

}}} // namespace lsst::afw::table
