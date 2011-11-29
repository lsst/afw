// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Layout_h_INCLUDED
#define AFW_TABLE_Layout_h_INCLUDED

#include "lsst/afw/table/config.h"

#include <set>

#include "boost/shared_ptr.hpp"

#include "lsst/ndarray.h"
#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/Field.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

class LayoutData;

} // namespace detail

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

    template <typename T>
    Key<T> add(Field<T> const & field);

    Layout();
    Layout(Layout const & other);
    
    Layout & operator=(Layout const & other);

    ~Layout();

private:

    friend class LayoutBuilder;
    friend class detail::Access;
    
    typedef detail::LayoutData Data;

    void finish();

    boost::shared_ptr<Data> _data;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Layout_h_INCLUDED
