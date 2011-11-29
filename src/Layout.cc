#include <list>
#include <stdexcept>

#include "boost/make_shared.hpp"
#include "boost/iterator/transform_iterator.hpp"
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/Layout.h"

namespace lsst { namespace afw { namespace table {

namespace {

struct Describe {

    template <typename T>
    void operator()(LayoutItem<T> const & item) const {
        result->insert(item.field.describe());
    }

    explicit Describe(Layout::Description * result_) : result(result_) {}

    Layout::Description * result;
};

template <typename T>
struct ExtractKey {

    typedef LayoutItem<T> argument_type;
    typedef Key<T> const & result_type;

    Key<T> const & operator()(LayoutItem<T> const & item) const {
        return item.key;
    }

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
    LayoutItem<T> item = { detail::Access::makeKey(field, _data->recordSize), field };
    _data->recordSize += field.getElementCount() * ELEMENT_SIZE;
    boost::fusion::at_key<T>(_data->items).push_back(item);
    return item.key;
}

Layout::Layout() : _data(boost::make_shared<Data>()) {}

template <typename T>
LayoutItem<T> Layout::find(std::string const & name) const {
    std::vector< LayoutItem<T> > const & vec = boost::fusion::at_key<T>(_data->items);
    for (typename std::vector< LayoutItem<T> >::const_iterator i = vec.begin(); i != vec.end(); ++i) {
        if (i->field.getName() == name) return *i;
    }
    throw LSST_EXCEPT(
        lsst::pex::exceptions::NotFoundException,
        (boost::format("Field with name '%s' not found.") % name).str()
    );
}

template <typename T>
LayoutItem<T> Layout::find(Key<T> const & key) const {
    typedef std::vector< LayoutItem<T> > Vector;
    Vector const & vec = boost::fusion::at_key<T>(_data->items);
    typedef boost::transform_iterator<ExtractKey<T>,typename Vector::const_iterator> Iterator;
    Iterator i = std::lower_bound(Iterator(vec.begin()), Iterator(vec.end()), key);
    if (*i != key) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::NotFoundException,
            "Key not found in Layout."
        );
    }
    return *i.base();
}

template <typename T>
void Layout::replace(Key<T> const & key, Field<T> const & field) {
    typedef std::vector< LayoutItem<T> > Vector;
    typedef boost::transform_iterator<ExtractKey<T>,typename Vector::iterator> Iterator;
    Vector & vec = boost::fusion::at_key<T>(_data->items);
    Iterator i = std::lower_bound(Iterator(vec.begin()), Iterator(vec.end()), key);
    if (*i != key) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::NotFoundException,
            "Key not found in Layout."
        );
    }
    i.base()->field = field;
}

Layout::Description Layout::describe() const {
    Description result;
    _data->forEach(Describe(&result));
    return result;
}

void Layout::finish() {
    static int const MIN_RECORD_ALIGN = sizeof(double) * detail::LayoutData::ALIGN_N_DOUBLE;
    if (!_data.unique()) {
        boost::shared_ptr<Data> data(boost::make_shared<Data>(*_data));
        _data.swap(data);
    }
    _data->recordSize += (MIN_RECORD_ALIGN - _data->recordSize % MIN_RECORD_ALIGN);
}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_LAYOUT(r, data, elem)                               \
    template Key< elem > Layout::add(Field< elem > const &);            \
    template LayoutItem< elem > Layout::find(std::string const & ) const; \
    template LayoutItem< elem > Layout::find(Key< elem > const & ) const; \
    template void Layout::replace(Key< elem > const &, Field< elem > const &);

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_LAYOUT, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_FIELD_TYPE_N, AFW_TABLE_FIELD_TYPE_TUPLE)
)

}}} // namespace lsst::afw::table
