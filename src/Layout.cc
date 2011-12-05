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

    typedef void result_type;

    template <typename T>
    void operator()(LayoutItem<T> const & item) const {
        result->insert(item.field.describe());
    }

    explicit Describe(Layout::Description * result_) : result(result_) {}

    Layout::Description * result;
};

struct ExtractOffset : public boost::static_visitor<int> {

    typedef int result_type;

    template <typename T>
    result_type operator()(LayoutItem<T> const & item) const {
        return detail::Access::getOffset(item.key);
    }

    result_type operator()(detail::LayoutData::ItemVariant const & v) const {
        return boost::apply_visitor(*this, v);
    }

};

struct CompareName : public boost::static_visitor<bool> {
    
    typedef bool result_type;
    
    template <typename T>
    result_type operator()(LayoutItem<T> const & item) const {
        return item.field.getName() == _name;
    }

    result_type operator()(detail::LayoutData::ItemVariant const & v) const {
        return boost::apply_visitor(*this, v);
    }

    explicit CompareName(std::string const & name) : _name(name) {}

    std::string _name;
};

template <typename T, typename VariantIterator>
LayoutItem<T> & findByOffset(int offset, VariantIterator const begin, VariantIterator const end) {
    typedef boost::transform_iterator<ExtractOffset,VariantIterator> Iterator;
    Iterator i = std::lower_bound(
        Iterator(begin),
        Iterator(end), 
        offset
    );
    if (*i != offset) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::NotFoundException,
            "Key not found in Layout."
        );
    }
    try {
        return boost::get< LayoutItem<T> >(*i.base());
    } catch (boost::bad_get & err) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            "Field with the given key offset does not have the given type."
        );
    }
};


} // anonymous

void Layout::_edit() {
    if (!_data.unique()) {
        boost::shared_ptr<Data> data(boost::make_shared<Data>(*_data));
        _data.swap(data);
    }
}

template <typename T>
Key<T> Layout::addField(Field<T> const & field) {
    static int const ELEMENT_SIZE = sizeof(typename Field<T>::Element);
    _edit();
    int padding = ELEMENT_SIZE - _data->_recordSize % ELEMENT_SIZE;
    if (padding != ELEMENT_SIZE) {
        _data->_recordSize += padding;
    }
    LayoutItem<T> item = { detail::Access::makeKey(field, _data->_recordSize), field };
    _data->_recordSize += field.getElementCount() * ELEMENT_SIZE;
    _data->_items.push_back(item);
    return item.key;
}

Layout::Layout() : _data(boost::make_shared<Data>()) {}

template <typename T>
LayoutItem<T> Layout::find(std::string const & name) const {
    Data::ItemContainer::iterator i = std::find_if(
        _data->_items.begin(),
        _data->_items.end(),
        CompareName(name)
    );
    if (i == _data->_items.end()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::NotFoundException,
            (boost::format("Field with name '%s' not found.") % name).str()
        );
    }
    try {
        return boost::get< LayoutItem<T> >(*i);
    } catch (boost::bad_get & err) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Field with name '%s' does not have the given type.") % name).str()
        );
    }
}

template <typename T>
LayoutItem<T> Layout::find(Key<T> const & key) const {
    return findByOffset<T>(
        detail::Access::getOffset(key),
        _data->_items.begin(),
        _data->_items.end()
    );
}

template <typename T>
void Layout::replaceField(Key<T> const & key, Field<T> const & field) {
    _edit();
    LayoutItem<T> & item = findByOffset<T>(
        detail::Access::getOffset(key),
        _data->_items.begin(),
        _data->_items.end()
    );
    item.field = field;
}

Layout::Description Layout::describe() const {
    Description result;
    forEach(Describe(&result));
    return result;
}

//----- Explicit instantiation ------------------------------------------------------------------------------

#define INSTANTIATE_LAYOUT(r, data, elem)                               \
    template Key< elem > Layout::addField(Field< elem > const &);            \
    template LayoutItem< elem > Layout::find(std::string const & ) const; \
    template LayoutItem< elem > Layout::find(Key< elem > const & ) const; \
    template void Layout::replaceField(Key< elem > const &, Field< elem > const &);

BOOST_PP_SEQ_FOR_EACH(
    INSTANTIATE_LAYOUT, _,
    BOOST_PP_TUPLE_TO_SEQ(AFW_TABLE_FIELD_TYPE_N, AFW_TABLE_FIELD_TYPE_TUPLE)
)

}}} // namespace lsst::afw::table
