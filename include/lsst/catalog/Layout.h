// -*- c++ -*-
#ifndef CATALOG_Layout_h_INCLUDED
#define CATALOG_Layout_h_INCLUDED

#include "lsst/catalog/detail/fusion_limits.h"

#include <set>

#include "boost/shared_ptr.hpp"

#include "lsst/ndarray.h"
#include "lsst/catalog/Key.h"
#include "lsst/catalog/Field.h"

namespace lsst { namespace catalog {

namespace detail {

class LayoutData;

} // namespace detail

class Layout;

class LayoutBuilder {
public:

    template <typename T>
    Key<T> add(Field<T> const & field);

    Layout finish();

    LayoutBuilder();
    LayoutBuilder(LayoutBuilder const & other);
    
    LayoutBuilder & operator=(LayoutBuilder const & other);

    ~LayoutBuilder();

private:

    friend class Layout;
    
    typedef detail::LayoutData Data;

    boost::shared_ptr<Data> _data;
};

class Layout {
public:

    typedef std::set<FieldDescription> Description;

    template <typename T>
    struct Item {
        Key<T> key;
        Field<T> field;
    };

    template <typename T>
    Item<T> find(std::string const & name) const;

    Description describe() const;

    int getRecordSize() const;

    ~Layout();

private:

    friend class LayoutBuilder;
    friend class detail::Access;
    
    typedef detail::LayoutData Data;

    Layout(boost::shared_ptr<Data> const & data);

    boost::shared_ptr<Data> _data;
};

}} // namespace lsst::catalog

#endif // !CATALOG_Layout_h_INCLUDED
