#include <list>
#include <stdexcept>

#include "boost/make_shared.hpp"
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/catalog/Layout.h"
#include "lsst/catalog/detail/Access.h"
#include "lsst/catalog/detail/LayoutData.h"

namespace lsst { namespace catalog {

//----- LayoutBuilder implementation ------------------------------------------------------------------------

template <typename T>
Key<T> LayoutBuilder::add(Field<T> const & field) {
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

Layout LayoutBuilder::finish() {
    static int const MIN_RECORD_ALIGN = sizeof(double) * detail::LayoutData::ALIGN_N_DOUBLE;
    if (!_data.unique()) {
        boost::shared_ptr<Data> data(boost::make_shared<Data>(*_data));
        _data.swap(data);
    }
    _data->recordSize += (MIN_RECORD_ALIGN - _data->recordSize % MIN_RECORD_ALIGN);
    return _data;
}

LayoutBuilder::LayoutBuilder() : _data(boost::make_shared<Data>()) {}

LayoutBuilder::LayoutBuilder(LayoutBuilder const & other) : _data(other._data) {}

LayoutBuilder & LayoutBuilder::operator=(LayoutBuilder const & other) {
    _data = other._data;
    return *this;
}

LayoutBuilder::~LayoutBuilder() {}

//----- Layout implementation -------------------------------------------------------------------------------

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

Layout::Description Layout::describe() const {
    Description result;
    _data->forEachItem(Describe(&result));
    return result;
}

int Layout::getRecordSize() const {
    return _data->recordSize;
}

Layout::Layout(boost::shared_ptr<Layout::Data> const & data) : _data(data) {}

Layout::~Layout() {}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_LAYOUT(r, data, elem)                           \
    template Key< elem > LayoutBuilder::add(Field< elem > const &); \
    template Layout::Item< elem > Layout::find(std::string const & ) const;

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_LAYOUT, _,
    BOOST_PP_TUPLE_TO_SEQ(CATALOG_FIELD_TYPE_N, CATALOG_FIELD_TYPE_TUPLE)
)

}} // namespace lsst::catalog
