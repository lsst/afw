#include "boost/fusion/algorithm/iteration/for_each.hpp"

#include "catalog/Layout.h"

namespace catalog {

int Layout::findOffset(int size, int align) {
    for (std::list<Gap>::iterator i = _gaps.begin(); i != _gaps.end(); ++i) {
        if (i->offset % align == 0 && i->size >= size) {
            int offset = i->offset;
            if (i->size == size) {
                _gaps.erase(i);
            } else {
                i->offset += size;
                i->size -= size;
            }
            return offset;
        }
    }
    int extra = align - _bytes % align;
    if (extra == align) {
        int offset = _bytes;
        _bytes += size;
        return offset;
    } else {
        Gap gap = { _bytes, extra };
        _bytes += extra;
        _gaps.push_back(gap);
        int offset = _bytes;
        _bytes += size;
        return offset;
    }
}

struct Layout::Describe {

    template <typename T>
    void operator()(boost::fusion::pair< T, std::vector< Item<T> > > const & type) const {
        for (
             typename std::vector< Item<T> >::const_iterator i = type.second.begin();
             i != type.second.end();
             ++i
        ) {
            result->insert(i->field.describe());
        }
    }

    Description * result;
};

Layout::Description Layout::describe() const {
    Description result;
    Describe f = { &result };
    boost::fusion::for_each(_data, f);
    return result;
}

} // namespace catalog
