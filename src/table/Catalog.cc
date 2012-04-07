#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/Simple.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/BaseTable.h"

namespace lsst { namespace afw { namespace table {

#if 0 // confuse emacs's dumb indenter
        }}}
#endif
template <typename RecordT>
CatalogT<RecordT> CatalogT<RecordT>::subset(
    std::ptrdiff_t startd, std::ptrdiff_t stopd, std::ptrdiff_t step) const {
    size_type S = size();
    size_type start, stop;
    if (startd < 0)
        startd += S;
    if (stopd  < 0)
        stopd  += S;
    if (startd < 0)
        startd = 0;
    start = (size_type)startd;
    if (start > S)
        start = S;
    
    if (step > 0) {
        if (stopd < 0)
            stopd = 0;
        stop = (size_type)stopd;
        if (stop > S)
            stop = S;
    } else if (step < 0) {
        if (stopd < 0)
            stopd = -1;
    }

    if (((step > 0) && (start >= stop)) ||
        ((step < 0) && ((std::ptrdiff_t)start <= stopd))) {
        // Empty
        return CatalogT<RecordT>(getTable(), begin(), begin());
    }

    if (step == 1) {
        assert(start >= 0);
        assert(stop  >  0);
        assert(start <  S);
        assert(stop  <= S);
        //std::cerr << "  subset: [" << start << ", " << stop << ")\n";
        return CatalogT<RecordT>(getTable(), begin()+start, begin()+stop);
    }
    // Build a new CatalogT and copy records into it.
    CatalogT<RecordT> cat(getTable());
    size_type N = 0;
    if (step >= 0)
        for (size_type i=start; i<stop; i+=step)
            N++;
    else {
        std::cerr << "subset: start=" << start << ", stopd=" << stopd << ", step=" << step << "\n";
        for (std::ptrdiff_t i=(std::ptrdiff_t)start; i>stopd; i+=step) {
            std::cerr << "  i = " << i << "\n";
            N++;
        }
    }
    cat.reserve(N);
    if (step >= 0)
        for (size_type i=start; i<stop; i+=step)
            cat.push_back(get(i));
    else {
        for (std::ptrdiff_t i=(std::ptrdiff_t)start; i>stopd; i+=step)
            cat.push_back(get(i));
    }
    return cat;
}

#if 0 // confuse emacs's dumb indenter
{{{
#endif

template class CatalogT<SimpleRecord>;
template class CatalogT<SimpleRecord const>;

template class SimpleCatalogT<SimpleRecord>;
template class SimpleCatalogT<SimpleRecord const>;

template class CatalogT<SourceRecord>;
template class CatalogT<SourceRecord const>;

template class SimpleCatalogT<SourceRecord>;
template class SimpleCatalogT<SourceRecord const>;

template class CatalogT<BaseRecord>;
template class CatalogT<BaseRecord const>;

}}}
