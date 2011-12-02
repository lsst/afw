#include "lsst/afw/table/config.h"

#include <cassert>

#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/detail/RecordData.h"

namespace lsst { namespace afw { namespace table { namespace detail {

void setupPointers(RecordSet & records, RecordData * & back) {
    // These pointers optimize the case where children of a common parent have contiguous IDs.
    RecordData * parent = 0;
    RecordData * sibling = 0;
    for (RecordSet::iterator i = records.begin(); i != records.end(); ++i) {
        if (i->parentId) {
            if (!parent || parent->id != i->parentId) {
                if (i->parentId >= i->id) {
                    throw LSST_EXCEPT(
                        lsst::pex::exceptions::LogicErrorException,
                        (boost::format(
                            "All child records must have IDs strictly greater than their parents; "
                            "%lld >= %lld"
                        ) % i->parentId % i->id).str()
                    );
                }
                RecordSet::iterator p = records.find(i->parentId, CompareRecordIdLess());                
                if (p == records.end()) {
                    throw LSST_EXCEPT(
                        lsst::pex::exceptions::NotFoundException,
                        (boost::format(
                            "Parent record %lld not found for child %lld."
                        ) % i->parentId % i->id).str()
                    );
                }
                parent = &(*p);
                sibling = 0;
            }
            i->links.initialize();
            if (!parent->links.child) {
                parent->links.child = &(*i);
            } else {
                if (!sibling) {
                    sibling = parent->links.child;
                    while (sibling->links.next) {
                        sibling = sibling->links.next;
                    }
                }
                sibling->links.next = &(*i);
                i->links.previous = sibling;
            }
            i->links.parent = parent;
            sibling = &(*i);
        } else { // no parent
            i->links.initialize();
            i->links.parent = 0;
            i->links.previous = back;
            if (back) {
                back->links.next = &(*i);
            }
            back = &(*i);
        }
    }
}

}}}} // namespace lsst::afw::table::detail
