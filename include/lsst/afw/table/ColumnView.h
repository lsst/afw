// -*- lsst-c++ -*-
#ifndef AFW_TABLE_ColumnView_h_INCLUDED
#define AFW_TABLE_ColumnView_h_INCLUDED

#include "lsst/afw/table/Schema.h"

namespace lsst { namespace afw { namespace table {

class TableBase;

/**
 *  @brief Column-wise view into a consolidated table.
 */
class ColumnView {
public:

    /// @brief Return the schema that defines the fields.
    Schema getSchema() const;

    /// @brief Return a 1-d array corresponding to a scalar field (or subfield).
    template <typename T>
    typename ndarray::Array<T const,1> operator[](Key<T> const & key) const;

    /// @brief Return a 2-d array corresponding to an array field.
    template <typename T>
    typename ndarray::Array<T const,2,1> operator[](Key< Array<T> > const & key) const;

    ~ColumnView();

private:

    friend class TableBase;

    struct Impl;

    ColumnView(Schema const & schema, int recordCount, void * buf, ndarray::Manager::Ptr const & manager);

    boost::shared_ptr<Impl> _impl;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_ColumnView_h_INCLUDED
