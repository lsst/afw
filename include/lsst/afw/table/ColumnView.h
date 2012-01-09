// -*- lsst-c++ -*-
#ifndef AFW_TABLE_ColumnView_h_INCLUDED
#define AFW_TABLE_ColumnView_h_INCLUDED

#include "lsst/afw/table/Schema.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

/// Functor to compute a flag bit, used to create an ndarray expression template.
struct FlagBitExtractor {
    typedef Field<Flag>::Element argument_type;
    typedef bool result_type;

    result_type operator()(argument_type element) const { return element & _mask; }

    explicit FlagBitExtractor(Key<Flag> const & key) : _mask(argument_type(1) << key.getBit()) {}

private:
    argument_type _mask;
};

} // namespace detail

class TableBase;

/**
 *  @brief Column-wise view into a consolidated table.
 *
 *  A ColumnView can be constructed from a table using TableBase::getColumnView().
 *
 *  Geometric (point and shape) fields cannot be accessed through a ColumnView, but their
 *  scalar components can be.
 */
class ColumnView {
public:

    /// @brief Return the schema that defines the fields.
    Schema getSchema() const;

    /// @brief Return an array of record IDs.
    lsst::ndarray::Array<RecordId const,1> getId() const;

    /**
     *  @brief Return an array of parent IDs.
     *  
     *  Records with no parent will have a parent ID of zero.
     *  @throw lsst::pex::exceptions::LogicErrorException if !getSchema().hasTree().
     */
    lsst::ndarray::Array<RecordId const,1> getParentId() const;

    /// @brief Return a 1-d array corresponding to a scalar field (or subfield).
    template <typename T>
    typename ndarray::Array<T const,1> operator[](Key<T> const & key) const;

    /// @brief Return a 2-d array corresponding to an array field.
    template <typename T>
    typename ndarray::Array<T const,2,1> operator[](Key< Array<T> > const & key) const;

    /**
     *  @brief Return a 1-d array expression corresponding to a flag bit.
     *
     *  In C++, the return value is a lazy ndarray expression template that performs the bitwise
     *  AND on every element when that element is requested.  In Python,
     *  the result will be copied into a bool NumPy array.
     */
    ndarray::result_of::vectorize< detail::FlagBitExtractor,
                                   ndarray::Array< Field<Flag>::Element const,1> >::type
    operator[](Key<Flag> const & key) const;

    ~ColumnView();

private:

    friend class TableBase;

    struct Impl;

    ColumnView(Schema const & schema, int recordCount, void * buf, ndarray::Manager::Ptr const & manager);

    boost::shared_ptr<Impl> _impl;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_ColumnView_h_INCLUDED
