#include <list>
#include <stdexcept>

#include "boost/make_shared.hpp"
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/detail/LayoutData.h"

namespace lsst { namespace afw { namespace table {

namespace {

struct Describe {

    template <typename T>
    void operator()(Layout::Item<T> const & item) const {
        result->insert(item.field.describe());
    }

    explicit Describe(Layout::Description * result_) : result(result_) {}

    Layout::Description * result;
};

} // anonymous

template <typename T>
Key<T> Layout::add(Field<T> const & field) {
    static int const ELEMENT_SIZE = sizeof(typename Field<T>::Element);
    if (!_data.unique()) {
        boost::shared_ptr<Data> data(boost::make_shared<Data>(*_data));
        _data.swap(data);
    }
    int padding = ELEMENT_SIZE - _data->recordSize % ELEMENT_SIZE;
    if (padding != ELEMENT_SIZE) {
        _data->recordSize += padding;
    }
    Layout::Item<T> item = { detail::Access::makeKey(field, _data->recordSize), field };
    _data->recordSize += field.getElementCount() * ELEMENT_SIZE;
    boost::fusion::at_key<T>(_data->items).push_back(item);
    return item.key;
}

Layout::Layout() : _data(boost::make_shared<Data>()) {}

Layout::Layout(Layout const & other) : _data(other._data) {}

Layout & Layout::operator=(Layout const & other) {
    _data = other._data;
    return *this;
}

template <typename T>
Layout::Item<T> Layout::find(std::string const & name) const {
    std::vector< Item<T> > const & vec = boost::fusion::at_key<T>(_data->items);
    for (typename std::vector< Item<T> >::const_iterator i = vec.begin(); i != vec.end(); ++i) {
        if (i->field.getName() == name) return *i;
    }
    throw LSST_EXCEPT(
        lsst::pex::exceptions::NotFoundException,
        (boost::format("Field with name '%s' not found.") % name).str()
    );
}

Layout::Description Layout::describe() const {
    Description result;
    _data->forEachItem(Describe(&result));
    return result;
}

int Layout::getRecordSize() const {
    return _data->recordSize;
}

void Layout::finish() {
    static int const MIN_RECORD_ALIGN = sizeof(double) * detail::LayoutData::ALIGN_N_DOUBLE;
    if (!_data.unique()) {
        boost::shared_ptr<Data> data(boost::make_shared<Data>(*_data));
        _data.swap(data);
    }
    _data->recordSize += (MIN_RECORD_ALIGN - _data->recordSize % MIN_RECORD_ALIGN);
}

Layout::~Layout() {}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_LAYOUT(r, data, elem)                           \
    template Key< elem > Layout::add(Field< elem > const &); \
    template Layout::Item< elem > Layout::find(std::string const & ) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_LAYOUT, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_FIELD_TYPE_N, AFW_TABLE_FIELD_TYPE_TUPLE)
)

}}} // namespace lsst::afw::table
