// -*- c++ -*-
#ifndef CATALOG_Layout_h_INCLUDED
#define CATALOG_Layout_h_INCLUDED

#include <set>

#include "boost/shared_ptr.hpp"

#include "lsst/ndarray.h"
#include "lsst/catalog/Key.h"

namespace lsst { namespace catalog {

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

    class Impl;

    boost::shared_ptr<Impl> _impl;
};

class Layout {
public:

    typedef std::set<FieldDescription> Description;

    template <typename T>
    Key<T> find(std::string const & name) const;

    Description describe() const;

    int getRecordSize() const;

    ~Layout();

private:

    friend class LayoutBuilder;
    
    struct Data;

    Layout(boost::shared_ptr<Data> const & data);

    boost::shared_ptr<Data> _data;
};

}} // namespace lsst::catalog

#endif // !CATALOG_Layout_h_INCLUDED
